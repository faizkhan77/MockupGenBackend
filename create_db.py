import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Field, Relationship
from typing import List, Optional
import sqlalchemy

# --- Pydantic Models for Database Tables ---

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str

    # This creates the one-to-many relationship
    # The 'images' attribute on a User object will be a list of MockupImage objects.
    images: List["MockupImage"] = Relationship(back_populates="user")


class MockupImage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    # Use LONGTEXT for storing base64 strings which can be very long
    base64_image: str = Field(sa_column=sqlalchemy.Column(sqlalchemy.LONGTEXT)) 
    
    # This creates the many-to-one relationship
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    user: Optional[User] = Relationship(back_populates="images")


# --- Database Creation Logic ---

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables.")

# The database name is the part after the last '/'
DB_NAME = DATABASE_URL.split("/")[-1]

# Create a connection to MySQL server (without specifying a database)
# This is necessary to be able to create the database itself.
engine_url_no_db = DATABASE_URL.rsplit('/', 1)[0]
engine_no_db = create_engine(engine_url_no_db)

# Create the database if it doesn't exist
with engine_no_db.connect() as connection:
    connection.execute(sqlalchemy.text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
    print(f"Database '{DB_NAME}' created or already exists.")

# Now connect to the specific database to create tables
engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    print("Creating tables...")
    # This command creates all tables defined by SQLModel subclasses
    SQLModel.metadata.create_all(engine)
    print("Tables created successfully.")

if __name__ == "__main__":
    create_db_and_tables()