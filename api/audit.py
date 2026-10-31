"""
Audit Trail
=============
Persists every transaction decision for traceability and compliance.

Architecture decision: Two backends
  - LOCAL (SQLite):    Zero setup, used for dev/demo. Data in audit_trail.db
  - PRODUCTION (DynamoDB): AWS free tier, used when AWS credentials are set.

Why not just use a log file?
  - Need to QUERY by transaction_id, status, time range
  - Need to UPDATE status (PENDING → APPROVED) atomically
  - Dashboard needs fast reads with filtering
  - Audit compliance requires tamper-evident, structured records
"""

import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from api.schemas import AuditRecord, ReviewStatus


class AuditTrail:
    """Base interface for audit trail operations."""

    def record_decision(self, record: AuditRecord) -> None:
        raise NotImplementedError

    def update_review(self, transaction_id: str, status: str,
                      reviewer_id: str = "", notes: str = "") -> bool:
        raise NotImplementedError

    def get_transaction(self, transaction_id: str) -> Optional[dict]:
        raise NotImplementedError

    def get_pending_reviews(self, limit: int = 50) -> list[dict]:
        raise NotImplementedError

    def get_recent(self, limit: int = 100) -> list[dict]:
        raise NotImplementedError

    def get_stats(self) -> dict:
        raise NotImplementedError


class SQLiteAuditTrail(AuditTrail):
    """
    Local SQLite backend — zero dependencies, works out of the box.
    Perfect for development, demos, and single-instance deployment.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "audit_trail.db")
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                transaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                amount REAL,
                merchant_category TEXT,
                fraud_probability REAL,
                risk_level TEXT,
                policy_decision TEXT,
                review_status TEXT DEFAULT 'PENDING',
                reviewer_id TEXT,
                reviewer_notes TEXT,
                investigation_summary TEXT,
                agent_trace_id TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_review_status
            ON audit_trail(review_status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON audit_trail(timestamp)
        """)
        conn.commit()
        conn.close()

    def record_decision(self, record: AuditRecord) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO audit_trail
               (transaction_id, timestamp, amount, merchant_category,
                fraud_probability, risk_level, policy_decision, review_status,
                reviewer_id, reviewer_notes, investigation_summary,
                agent_trace_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.transaction_id,
                record.timestamp,
                record.amount,
                record.merchant_category,
                record.fraud_probability,
                record.risk_level,
                record.policy_decision,
                record.review_status,
                record.reviewer_id,
                record.reviewer_notes,
                record.investigation_summary,
                record.agent_trace_id,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def update_review(self, transaction_id: str, status: str,
                      reviewer_id: str = "", notes: str = "") -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            """UPDATE audit_trail
               SET review_status = ?, reviewer_id = ?, reviewer_notes = ?,
                   updated_at = ?
               WHERE transaction_id = ?""",
            (status, reviewer_id, notes, datetime.utcnow().isoformat(),
             transaction_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated

    def update_investigation(self, transaction_id: str, summary: str,
                             trace_id: str = "") -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            """UPDATE audit_trail
               SET investigation_summary = ?, agent_trace_id = ?,
                   updated_at = ?
               WHERE transaction_id = ?""",
            (summary, trace_id, datetime.utcnow().isoformat(), transaction_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated

    def get_transaction(self, transaction_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM audit_trail WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_pending_reviews(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM audit_trail
               WHERE review_status = 'PENDING' AND policy_decision = 'INVESTIGATE'
               ORDER BY fraud_probability DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_recent(self, limit: int = 100) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM audit_trail ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM audit_trail").fetchone()[0]
        by_decision = {}
        for row in conn.execute(
            "SELECT policy_decision, COUNT(*) as cnt FROM audit_trail GROUP BY policy_decision"
        ).fetchall():
            by_decision[row["policy_decision"]] = row["cnt"]
        by_status = {}
        for row in conn.execute(
            "SELECT review_status, COUNT(*) as cnt FROM audit_trail GROUP BY review_status"
        ).fetchall():
            by_status[row["review_status"]] = row["cnt"]
        avg_score = conn.execute(
            "SELECT AVG(fraud_probability) FROM audit_trail"
        ).fetchone()[0]
        conn.close()

        return {
            "total_transactions": total,
            "by_decision": by_decision,
            "by_review_status": by_status,
            "avg_fraud_probability": round(avg_score, 4) if avg_score else 0,
        }


class DynamoDBAuditTrail(AuditTrail):
    """
    Production backend using AWS DynamoDB.
    Free tier: 25GB storage, 25 RCU/WCU (always free, not 12-month).

    Table design:
      PK: transaction_id
      GSI1: review_status + timestamp (for pending reviews query)
      GSI2: policy_decision + timestamp (for decision analytics)
    """

    def __init__(self, table_name: str = "fraud_audit_trail", region: str = "us-east-1"):
        import boto3
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.table_name = table_name
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.table = self.dynamodb.Table(self.table_name)
            self.table.load()
        except Exception:
            self.table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {"AttributeName": "transaction_id", "KeyType": "HASH"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "transaction_id", "AttributeType": "S"},
                    {"AttributeName": "review_status", "AttributeType": "S"},
                    {"AttributeName": "timestamp", "AttributeType": "S"},
                ],
                GlobalSecondaryIndexes=[
                    {
                        "IndexName": "status-time-index",
                        "KeySchema": [
                            {"AttributeName": "review_status", "KeyType": "HASH"},
                            {"AttributeName": "timestamp", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                        "ProvisionedThroughput": {
                            "ReadCapacityUnits": 5,
                            "WriteCapacityUnits": 5,
                        },
                    },
                ],
                ProvisionedThroughput={
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5,
                },
            )
            self.table.wait_until_exists()

    def record_decision(self, record: AuditRecord) -> None:
        from decimal import Decimal
        item = json.loads(record.model_dump_json())
        # DynamoDB needs Decimal instead of float
        for k, v in item.items():
            if isinstance(v, float):
                item[k] = Decimal(str(v))
        item["updated_at"] = datetime.utcnow().isoformat()
        self.table.put_item(Item=item)

    def update_review(self, transaction_id: str, status: str,
                      reviewer_id: str = "", notes: str = "") -> bool:
        try:
            self.table.update_item(
                Key={"transaction_id": transaction_id},
                UpdateExpression="SET review_status = :s, reviewer_id = :r, "
                                 "reviewer_notes = :n, updated_at = :u",
                ExpressionAttributeValues={
                    ":s": status,
                    ":r": reviewer_id,
                    ":n": notes,
                    ":u": datetime.utcnow().isoformat(),
                },
            )
            return True
        except Exception:
            return False

    def get_transaction(self, transaction_id: str) -> Optional[dict]:
        resp = self.table.get_item(Key={"transaction_id": transaction_id})
        item = resp.get("Item")
        if item:
            return {k: float(v) if hasattr(v, "as_tuple") else v for k, v in item.items()}
        return None

    def get_pending_reviews(self, limit: int = 50) -> list[dict]:
        resp = self.table.query(
            IndexName="status-time-index",
            KeyConditionExpression="review_status = :s",
            ExpressionAttributeValues={":s": "PENDING"},
            ScanIndexForward=False,
            Limit=limit,
        )
        return resp.get("Items", [])

    def get_recent(self, limit: int = 100) -> list[dict]:
        resp = self.table.scan(Limit=limit)
        items = resp.get("Items", [])
        return sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_stats(self) -> dict:
        resp = self.table.scan(Select="COUNT")
        return {"total_transactions": resp["Count"]}


def get_audit_trail() -> AuditTrail:
    """
    Factory: returns DynamoDB in production, SQLite for local dev.
    Switches based on whether AWS credentials are configured.
    """
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("USE_DYNAMODB", "").lower() == "true":
        return DynamoDBAuditTrail(
            table_name=os.getenv("DYNAMODB_TABLE", "fraud_audit_trail"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
    return SQLiteAuditTrail()
