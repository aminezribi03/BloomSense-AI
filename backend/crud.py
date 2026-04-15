
import sqlite3
from typing import Dict, List


def create_prediction(
    conn: sqlite3.Connection,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    predicted_class: int,
    probability: float,
) -> None:
    """Insert a new prediction record into the database."""
    conn.execute(

        (sepal_length, sepal_width, petal_length, petal_width, predicted_class, probability),
    )


def get_history(conn: sqlite3.Connection) -> List[Dict[str, int]]:
   
    cursor = conn.execute(
        
    )
    rows = cursor.fetchall()
    return [
        {"class_label": str(row[0]), "count": row[1]}
        for row in rows
    ]
