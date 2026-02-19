"""
Chat History Database
SQLite database for persisting chat history
"""

import sqlite3
import logging
import time
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ChatHistoryDB:
    """SQLite database for chat history"""
    
    def __init__(self, db_path: str = "data/history.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT DEFAULT 'localuser',
                message TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                timestamp INTEGER NOT NULL,
                context_ref TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_message(self, session_id: str, user_id: str, message: str, role: str, context_ref: Optional[str] = None, metadata: Optional[str] = None) -> int:
        """Save a message to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO history (session_id, user_id, message, role, timestamp, context_ref, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, user_id, message, role, int(time.time()), context_ref, metadata))
        
        message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_history(self, session_id: str, user_id: Optional[str] = None) -> List[Dict]:
        """Get chat history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute("""
                SELECT id, message, role, timestamp, context_ref, metadata
                FROM history
                WHERE session_id = ? AND user_id = ?
                ORDER BY timestamp ASC
            """, (session_id, user_id))
        else:
            cursor.execute("""
                SELECT id, message, role, timestamp, context_ref, metadata
                FROM history
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "message": row[1],
                "role": row[2],
                "timestamp": row[3],
                "context_ref": row[4],
                "metadata": row[5]
            }
            for row in rows
        ]
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """List all chat sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute("""
                SELECT DISTINCT session_id, MAX(timestamp) as last_message_time, COUNT(*) as message_count
                FROM history
                WHERE user_id = ?
                GROUP BY session_id
                ORDER BY last_message_time DESC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT DISTINCT session_id, MAX(timestamp) as last_message_time, COUNT(*) as message_count
                FROM history
                GROUP BY session_id
                ORDER BY last_message_time DESC
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "session_id": row[0],
                "last_message_time": row[1],
                "message_count": row[2]
            }
            for row in rows
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted_count > 0



