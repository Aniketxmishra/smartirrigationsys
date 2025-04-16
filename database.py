import sqlite3
from datetime import datetime
import json

class IrrigationDatabase:
    def __init__(self, db_name='irrigation_system.db'):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.initialize_database()

    def connect(self):
        """Establish connection to the database"""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def initialize_database(self):
        """Create database tables if they don't exist"""
        self.connect()
        
        # System configuration table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            num_zones INTEGER NOT NULL,
            reservoir_capacity REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Zones table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_number INTEGER NOT NULL,
            area REAL NOT NULL,
            soil_type TEXT NOT NULL,
            crop_type TEXT NOT NULL,
            moisture_threshold REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(zone_number)
        )
        ''')

        # Weather conditions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_conditions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            rainfall REAL NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Irrigation events table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS irrigation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            water_amount REAL NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            status TEXT NOT NULL,
            FOREIGN KEY (zone_id) REFERENCES zones (id)
        )
        ''')

        # Moisture readings table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS moisture_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            moisture_level REAL NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (zone_id) REFERENCES zones (id)
        )
        ''')

        # System status table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reservoir_level REAL NOT NULL,
            system_state TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.conn.commit()
        self.close()

    def save_system_config(self, num_zones, reservoir_capacity):
        """Save system configuration"""
        self.connect()
        self.cursor.execute('''
        INSERT INTO system_config (num_zones, reservoir_capacity)
        VALUES (?, ?)
        ''', (num_zones, reservoir_capacity))
        self.conn.commit()
        self.close()

    def save_zone(self, zone_number, area, soil_type, crop_type, moisture_threshold):
        """Save zone information"""
        self.connect()
        try:
            self.cursor.execute('''
            INSERT INTO zones (zone_number, area, soil_type, crop_type, moisture_threshold)
            VALUES (?, ?, ?, ?, ?)
            ''', (zone_number, area, soil_type, crop_type, moisture_threshold))
            self.conn.commit()
        except sqlite3.IntegrityError:
            # If zone already exists, update it
            self.cursor.execute('''
            UPDATE zones 
            SET area = ?, soil_type = ?, crop_type = ?, moisture_threshold = ?
            WHERE zone_number = ?
            ''', (area, soil_type, crop_type, moisture_threshold, zone_number))
            self.conn.commit()
        finally:
            self.close()

    def save_weather_condition(self, temperature, humidity, rainfall):
        """Save weather conditions"""
        self.connect()
        self.cursor.execute('''
        INSERT INTO weather_conditions (temperature, humidity, rainfall)
        VALUES (?, ?, ?)
        ''', (temperature, humidity, rainfall))
        self.conn.commit()
        self.close()

    def save_irrigation_event(self, zone_id, water_amount, start_time, status='completed'):
        """Save irrigation event"""
        self.connect()
        self.cursor.execute('''
        INSERT INTO irrigation_events (zone_id, water_amount, start_time, status)
        VALUES (?, ?, ?, ?)
        ''', (zone_id, water_amount, start_time, status))
        self.conn.commit()
        self.close()

    def save_moisture_reading(self, zone_id, moisture_level):
        """Save moisture reading"""
        self.connect()
        self.cursor.execute('''
        INSERT INTO moisture_readings (zone_id, moisture_level)
        VALUES (?, ?)
        ''', (zone_id, moisture_level))
        self.conn.commit()
        self.close()

    def update_system_status(self, reservoir_level, system_state):
        """Update system status"""
        self.connect()
        self.cursor.execute('''
        INSERT INTO system_status (reservoir_level, system_state)
        VALUES (?, ?)
        ''', (reservoir_level, system_state))
        self.conn.commit()
        self.close()

    def get_latest_weather_conditions(self):
        """Get latest weather conditions"""
        self.connect()
        self.cursor.execute('''
        SELECT temperature, humidity, rainfall, recorded_at
        FROM weather_conditions
        ORDER BY recorded_at DESC
        LIMIT 1
        ''')
        result = self.cursor.fetchone()
        self.close()
        return result

    def get_zone_moisture_history(self, zone_id, limit=24):
        """Get moisture history for a zone"""
        self.connect()
        self.cursor.execute('''
        SELECT moisture_level, recorded_at
        FROM moisture_readings
        WHERE zone_id = ?
        ORDER BY recorded_at DESC
        LIMIT ?
        ''', (zone_id, limit))
        results = self.cursor.fetchall()
        self.close()
        return results

    def get_irrigation_history(self, zone_id=None, limit=10):
        """Get irrigation history"""
        self.connect()
        if zone_id:
            self.cursor.execute('''
            SELECT i.*, z.zone_number
            FROM irrigation_events i
            JOIN zones z ON i.zone_id = z.id
            WHERE i.zone_id = ?
            ORDER BY i.start_time DESC
            LIMIT ?
            ''', (zone_id, limit))
        else:
            self.cursor.execute('''
            SELECT i.*, z.zone_number
            FROM irrigation_events i
            JOIN zones z ON i.zone_id = z.id
            ORDER BY i.start_time DESC
            LIMIT ?
            ''', (limit,))
        results = self.cursor.fetchall()
        self.close()
        return results

    def get_system_status(self):
        """Get current system status"""
        self.connect()
        self.cursor.execute('''
        SELECT reservoir_level, system_state, last_updated
        FROM system_status
        ORDER BY last_updated DESC
        LIMIT 1
        ''')
        result = self.cursor.fetchone()
        self.close()
        if result:
            reservoir_level, system_state, last_updated = result
            try:
                # Try to parse the timestamp string
                last_updated_dt = datetime.strptime(last_updated, '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                # If parsing fails, use current time
                last_updated_dt = datetime.now()
            return reservoir_level, system_state, last_updated_dt
        return None

    def clear_zones(self):
        """Clear all existing zones"""
        self.connect()
        self.cursor.execute('DELETE FROM zones')
        self.cursor.execute('DELETE FROM moisture_readings')
        self.cursor.execute('DELETE FROM irrigation_events')
        self.conn.commit()
        self.close() 