import asyncio
import os
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine, text, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Database connection parameters - Updated to use the password from hypha-secrets
DB_USER = "hypha-admin"
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = "hypha-db"
DB_HOST = "localhost"  # Using localhost with port-forwarding 
DB_PORT = "5432"       # Using standard PostgreSQL port

# Create connection URL
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(f"Connecting to: {DATABASE_URL}")

# PostgreSQL engine and session
pg_engine = create_async_engine(DATABASE_URL, echo=False)  # Set echo to False to reduce output
pg_session_maker = async_sessionmaker(pg_engine, expire_on_commit=False, class_=AsyncSession)

async def check_database_stats():
    """Query the database and display table statistics without writing data"""
    print("🔍 Checking PostgreSQL database status...")
    
    try:
        async with pg_session_maker() as session:
            # Get list of all tables
            result = await session.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = result.scalars().all()
            
            print(f"📊 Database contains {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
                
                # Get count of rows per table
                count_query = text(f"SELECT COUNT(*) FROM {table}")
                count_result = await session.execute(count_query)
                row_count = count_result.scalar()
                print(f"    • Rows: {row_count}")
                
                # Get table size
                size_query = text(f"SELECT pg_size_pretty(pg_total_relation_size('{table}'))")
                size_result = await session.execute(size_query)
                table_size = size_result.scalar()
                print(f"    • Size: {table_size}")
                
                # If it's the artifacts table and it exists, show some stats
                if table == 'artifacts':
                    # Count by workspace
                    try:
                        workspace_query = text("SELECT workspace, COUNT(*) FROM artifacts GROUP BY workspace")
                        workspace_result = await session.execute(workspace_query)
                        workspace_counts = workspace_result.all()
                        
                        if workspace_counts:
                            print(f"    • Artifacts by workspace:")
                            for workspace, count in workspace_counts:
                                print(f"      - {workspace}: {count} artifacts")
                    except Exception as e:
                        print(f"    • Could not get workspace stats: {str(e)}")
                
            # Get database size
            db_size_query = text("SELECT pg_size_pretty(pg_database_size(current_database()))")
            db_size_result = await session.execute(db_size_query)
            db_size = db_size_result.scalar()
            print(f"\n💾 Total database size: {db_size}")
            
            # Get connection info
            version_query = text("SELECT version()")
            version_result = await session.execute(version_query)
            version = version_result.scalar()
            print(f"\n🔌 Connected to: {version}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error checking database stats: {str(e)}")
        try_sync_connection()
        return False

async def test_database_write_read():
    """Test database by creating a temporary table, writing data, reading it back, and dropping the table"""
    print("\n🧪 Testing database write/read operations...")
    
    test_table_name = "test_hypha_table"
    
    try:
        async with pg_session_maker() as session:
            # Check if test table exists and drop it to ensure clean test
            check_table = text(f"SELECT to_regclass('public.{test_table_name}')")
            result = await session.execute(check_table)
            if result.scalar():
                print(f"  • Found existing test table, dropping it first")
                await session.execute(text(f"DROP TABLE {test_table_name}"))
                await session.commit()
            
            # Create test table
            create_table = text(f"""
                CREATE TABLE {test_table_name} (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await session.execute(create_table)
            await session.commit()
            print(f"  • Created test table: {test_table_name}")
            
            # Insert test data
            test_data = [
                ("item1", 100),
                ("item2", 200),
                ("item3", 300)
            ]
            
            for name, value in test_data:
                insert = text(f"INSERT INTO {test_table_name} (name, value) VALUES (:name, :value)")
                await session.execute(insert, {"name": name, "value": value})
            
            await session.commit()
            print(f"  • Inserted {len(test_data)} rows of test data")
            
            # Read back data to verify
            select = text(f"SELECT id, name, value FROM {test_table_name} ORDER BY id")
            result = await session.execute(select)
            rows = result.fetchall()
            
            # Verify data integrity
            if len(rows) == len(test_data):
                print(f"  • Successfully read back {len(rows)} rows:")
                for row in rows:
                    print(f"    - ID: {row[0]}, Name: {row[1]}, Value: {row[2]}")
                
                # Verify values match what we inserted
                all_match = True
                for i, (name, value) in enumerate(test_data):
                    if rows[i][1] != name or rows[i][2] != value:
                        all_match = False
                        break
                
                if all_match:
                    print("  ✅ Data verification successful - all values match")
                else:
                    print("  ❌ Data verification failed - values don't match")
            else:
                print(f"  ❌ Data count mismatch: inserted {len(test_data)}, read back {len(rows)}")
            
            # Drop test table
            drop_table = text(f"DROP TABLE {test_table_name}")
            await session.execute(drop_table)
            await session.commit()
            print(f"  • Cleaned up: dropped test table {test_table_name}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error during database write/read test: {str(e)}")
        return False

def try_sync_connection():
    """Fallback to sync connection for troubleshooting"""
    try:
        print("\nTrying sync connection with psycopg2...")
        sync_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        # Create synchronous engine
        sync_engine = create_engine(sync_url)
        
        # Test connection
        with sync_engine.connect() as conn:
            tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            print(f"Tables in database: {[table[0] for table in tables]}")
            
            db_size = conn.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))"))
            print(f"Database size: {db_size.scalar()}")
            
            version = conn.execute(text("SELECT version()"))
            print(f"Connected to: {version.scalar()}")

            
        print("✅ Successfully connected with psycopg2!")
        
    except Exception as e2:
        print(f"❌ Error with sync connection: {str(e2)}")

async def main():
    print("🔄 Starting database connection test...")
    stats_success = await check_database_stats()
    
    if stats_success:
        # Run write/read test only if stats check was successful
        await test_database_write_read()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())