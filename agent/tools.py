from langchain_core.tools import Tool
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

DB_SCHEMA = """
books (id, title, isbn, published_year, category_id)
authors (id, name)
book_authors (book_id, author_id)
categories (id, name)
members (id, name, email, phone, membership_type_id, joined_at, expires_at)
membership_types (id, type_name, max_books, duration_days)
borrowings (id, book_id, member_id, borrowed_at, due_date, returned_at)
fines (id, borrowing_id, amount, paid, created_at)
librarians (id, name, email, role_id)
roles (id, role_name)
"""
@tool
def get_schema() -> str:
    """
    Returns the database schema for the library management system.
    """
    return DB_SCHEMA

tools = [get_schema]