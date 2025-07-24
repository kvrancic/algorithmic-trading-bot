"""
Database Schema Management for QuantumSentiment Trading Bot

Provides database initialization, migration, and session management utilities.
"""

import os
from typing import Generator
from contextlib import contextmanager
import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base
from .database import DatabaseManager

logger = structlog.get_logger(__name__)


def create_tables(database_url: str = None) -> bool:
    """
    Create all database tables
    
    Args:
        database_url: Database connection URL
        
    Returns:
        Success status
    """
    try:
        db_manager = DatabaseManager(database_url)
        db_manager.create_tables()
        return True
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        return False


def initialize_database(database_url: str = None) -> DatabaseManager:
    """
    Initialize database with all tables and return manager instance
    
    Args:
        database_url: Database connection URL
        
    Returns:
        DatabaseManager instance
    """
    try:
        db_manager = DatabaseManager(database_url)
        
        # Test connection
        if not db_manager.test_connection():
            raise Exception("Database connection test failed")
        
        # Create tables if they don't exist
        db_manager.create_tables()
        
        # Get initial stats
        stats = db_manager.get_database_stats()
        logger.info("Database initialized successfully", 
                   db_type=db_manager.db_type,
                   tables=len(stats))
        
        return db_manager
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


@contextmanager
def get_session(database_url: str = None) -> Generator[Session, None, None]:
    """
    Get database session with automatic cleanup
    
    Args:
        database_url: Database connection URL
        
    Yields:
        SQLAlchemy session
    """
    db_manager = DatabaseManager(database_url)
    
    try:
        with db_manager.get_session() as session:
            yield session
    finally:
        db_manager.engine.dispose()


def get_database_manager(database_url: str = None) -> DatabaseManager:
    """
    Get database manager instance with connection testing
    
    Args:
        database_url: Database connection URL
        
    Returns:
        DatabaseManager instance
    """
    db_manager = DatabaseManager(database_url)
    
    # Test connection on creation
    if not db_manager.test_connection():
        raise Exception("Failed to connect to database")
    
    return db_manager


def setup_sqlite_database(db_path: str = "data/quantum.db") -> str:
    """
    Setup SQLite database with proper directory structure
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Database URL
    """
    try:
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info("Created database directory", path=db_dir)
        
        # Create database URL
        database_url = f"sqlite:///{db_path}"
        
        # Initialize database
        db_manager = initialize_database(database_url)
        
        logger.info("SQLite database setup complete", path=db_path)
        return database_url
        
    except Exception as e:
        logger.error("Failed to setup SQLite database", 
                    path=db_path, error=str(e))
        raise


def setup_postgresql_database(
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
    **kwargs
) -> str:
    """
    Setup PostgreSQL database connection
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        username: Database username
        password: Database password
        **kwargs: Additional connection parameters
        
    Returns:
        Database URL
    """
    try:
        # Build connection URL
        database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        # Add additional parameters
        if kwargs:
            params = "&".join([f"{k}={v}" for k, v in kwargs.items()])
            database_url += f"?{params}"
        
        # Initialize database
        db_manager = initialize_database(database_url)
        
        logger.info("PostgreSQL database setup complete", 
                   host=host, 
                   port=port,
                   database=database)
        
        return database_url
        
    except Exception as e:
        logger.error("Failed to setup PostgreSQL database", 
                    host=host, 
                    database=database,
                    error=str(e))
        raise


def migrate_database(database_url: str = None) -> bool:
    """
    Perform database migrations (placeholder for future use)
    
    Args:
        database_url: Database connection URL
        
    Returns:
        Success status
    """
    try:
        # For now, just recreate tables
        # In the future, this would handle proper migrations
        db_manager = DatabaseManager(database_url)
        
        # Test connection
        if not db_manager.test_connection():
            return False
        
        # Create any missing tables
        db_manager.create_tables()
        
        logger.info("Database migration completed")
        return True
        
    except Exception as e:
        logger.error("Database migration failed", error=str(e))
        return False


def backup_database(database_url: str, backup_path: str) -> bool:
    """
    Backup database (SQLite only for now)
    
    Args:
        database_url: Source database URL
        backup_path: Backup file path
        
    Returns:
        Success status
    """
    try:
        if not database_url.startswith('sqlite'):
            logger.error("Database backup only supported for SQLite")
            return False
        
        import shutil
        
        # Extract source path from URL
        source_path = database_url.replace('sqlite:///', '')
        
        if not os.path.exists(source_path):
            logger.error("Source database file not found", path=source_path)
            return False
        
        # Ensure backup directory exists
        backup_dir = os.path.dirname(backup_path)
        if backup_dir and not os.path.exists(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)
        
        # Copy database file
        shutil.copy2(source_path, backup_path)
        
        logger.info("Database backup completed", 
                   source=source_path,
                   backup=backup_path)
        return True
        
    except Exception as e:
        logger.error("Database backup failed", 
                    backup_path=backup_path,
                    error=str(e))
        return False


def restore_database(backup_path: str, database_url: str) -> bool:
    """
    Restore database from backup (SQLite only)
    
    Args:
        backup_path: Backup file path
        database_url: Target database URL
        
    Returns:
        Success status
    """
    try:
        if not database_url.startswith('sqlite'):
            logger.error("Database restore only supported for SQLite")
            return False
        
        import shutil
        
        if not os.path.exists(backup_path):
            logger.error("Backup file not found", path=backup_path)
            return False
        
        # Extract target path from URL
        target_path = database_url.replace('sqlite:///', '')
        
        # Ensure target directory exists
        target_dir = os.path.dirname(target_path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        # Copy backup to target
        shutil.copy2(backup_path, target_path)
        
        # Test the restored database
        db_manager = DatabaseManager(database_url)
        if not db_manager.test_connection():
            logger.error("Restored database connection test failed")
            return False
        
        logger.info("Database restore completed", 
                   backup=backup_path,
                   target=target_path)
        return True
        
    except Exception as e:
        logger.error("Database restore failed", 
                    backup_path=backup_path,
                    error=str(e))
        return False


def get_database_info(database_url: str = None) -> dict:
    """
    Get comprehensive database information
    
    Args:
        database_url: Database connection URL
        
    Returns:
        Database information dictionary
    """
    try:
        db_manager = DatabaseManager(database_url)
        
        info = {
            'database_url': database_url or os.getenv('DATABASE_URL', 'sqlite:///data/quantum.db'),
            'database_type': db_manager.db_type,
            'connection_status': db_manager.test_connection(),
            'statistics': db_manager.get_database_stats()
        }
        
        # Add file size for SQLite
        if db_manager.db_type == 'sqlite':
            db_path = database_url.replace('sqlite:///', '') if database_url else 'data/quantum.db'
            if os.path.exists(db_path):
                info['file_size_bytes'] = os.path.getsize(db_path)
                info['file_size_mb'] = info['file_size_bytes'] / (1024 * 1024)
        
        return info
        
    except Exception as e:
        logger.error("Failed to get database info", error=str(e))
        return {
            'database_url': database_url,
            'connection_status': False,
            'error': str(e)
        }