#!/usr/bin/env python3
"""
Verify the RLVR demo database was created correctly
"""
import sqlite3
import os

def verify_database():
    db_path = "rlvr_demo.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    print(f"✅ Database found: {db_path}")
    print(f"📁 Size: {os.path.getsize(db_path)} bytes")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\n📊 Tables: {[table[0] for table in tables]}")
        
        # Check data counts
        for table_name, in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   {table_name}: {count} rows")
        
        # Show sample data
        print(f"\n🔍 Sample Employee Data:")
        cursor.execute("SELECT name, department, salary FROM employees LIMIT 3")
        for row in cursor.fetchall():
            print(f"   {row[0]} | {row[1]} | ${row[2]:,.0f}")
        
        print(f"\n🏢 Sample Project Data:")
        cursor.execute("SELECT name, status, budget FROM projects LIMIT 3")
        for row in cursor.fetchall():
            print(f"   {row[0]} | {row[1]} | ${row[2]:,.0f}")
        
        # Test a simple query
        print(f"\n🧪 Test Query: Engineering employees")
        cursor.execute("SELECT name, salary FROM employees WHERE department = 'Engineering'")
        eng_employees = cursor.fetchall()
        print(f"   Found {len(eng_employees)} Engineering employees:")
        for name, salary in eng_employees:
            print(f"   - {name}: ${salary:,.0f}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

if __name__ == "__main__":
    print("🎯 RLVR Database Verification")
    print("=" * 40)
    
    success = verify_database()
    
    if success:
        print(f"\n✅ Database verification successful!")
        print(f"💡 The RLVR demo database is ready for SQL testing")
    else:
        print(f"\n❌ Database verification failed!")