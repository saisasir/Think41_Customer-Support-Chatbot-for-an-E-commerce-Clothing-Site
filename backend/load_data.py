import pandas as pd
import os
from sqlalchemy.orm import sessionmaker
from database import engine, Base, create_tables
from models import User, Product, Order, OrderItem, InventoryItem, DistributionCenter
from datetime import datetime

# Create all tables
create_tables()

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def parse_datetime(date_str):
    """Parse datetime string with error handling"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def load_csv_data():
    """Load all CSV data into the database"""
    db = SessionLocal()
    
    try:
        print("🚀 Starting data ingestion...")
        
        # Load Distribution Centers
        print("📍 Loading distribution centers...")
        df_centers = pd.read_csv('../data/distribution_centers.csv')
        for _, row in df_centers.iterrows():
            center = DistributionCenter(
                id=row['id'],
                name=row['name'],
                latitude=row['latitude'],
                longitude=row['longitude']
            )
            db.merge(center)
        
        # Load Users
        print("👤 Loading users...")
        df_users = pd.read_csv('../data/users.csv')
        for _, row in df_users.iterrows():
            user = User(
                id=row['id'],
                first_name=row['first_name'],
                last_name=row['last_name'],
                email=row['email'],
                age=row['age'],
                gender=row['gender'],
                state=row['state'],
                street_address=row['street_address'],
                postal_code=row['postal_code'],
                city=row['city'],
                country=row['country'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                traffic_source=row['traffic_source'],
                created_at=parse_datetime(row['created_at'])
            )
            db.merge(user)
        
        # Load Products
        print("🛍️ Loading products...")
        df_products = pd.read_csv('../data/products.csv')
        for _, row in df_products.iterrows():
            product = Product(
                id=row['id'],
                cost=row['cost'],
                category=row['category'],
                name=row['name'],
                brand=row['brand'],
                retail_price=row['retail_price'],
                department=row['department'],
                sku=row['sku'],
                distribution_center_id=row['distribution_center_id']
            )
            db.merge(product)
        
        # Load Orders
        print("📋 Loading orders...")
        df_orders = pd.read_csv('../data/orders.csv')
        for _, row in df_orders.iterrows():
            order = Order(
                order_id=row['order_id'],
                user_id=row['user_id'],
                status=row['status'],
                gender=row['gender'],
                created_at=parse_datetime(row['created_at']),
                returned_at=parse_datetime(row['returned_at']),
                shipped_at=parse_datetime(row['shipped_at']),
                delivered_at=parse_datetime(row['delivered_at']),
                num_of_item=row['num_of_item']
            )
            db.merge(order)
        
        # Load Order Items
        print("🛒 Loading order items...")
        df_order_items = pd.read_csv('../data/order_items.csv')
        for _, row in df_order_items.iterrows():
            order_item = OrderItem(
                id=row['id'],
                order_id=row['order_id'],
                user_id=row['user_id'],
                product_id=row['product_id'],
                inventory_item_id=row['inventory_item_id'],
                status=row['status'],
                created_at=parse_datetime(row['created_at']),
                shipped_at=parse_datetime(row['shipped_at']),
                delivered_at=parse_datetime(row['delivered_at']),
                returned_at=parse_datetime(row['returned_at']),
                sale_price=row['sale_price']
            )
            db.merge(order_item)
        
        # Load Inventory Items
        print("📦 Loading inventory items...")
        df_inventory = pd.read_csv('../data/inventory_items.csv')
        for _, row in df_inventory.iterrows():
            inventory_item = InventoryItem(
                id=row['id'],
                product_id=row['product_id'],
                created_at=parse_datetime(row['created_at']),
                sold_at=parse_datetime(row['sold_at']),
                cost=row['cost'],
                product_category=row['product_category'],
                product_name=row['product_name'],
                product_brand=row['product_brand'],
                product_retail_price=row['product_retail_price'],
                product_department=row['product_department'],
                product_sku=row['product_sku'],
                product_distribution_center_id=row['product_distribution_center_id']
            )
            db.merge(inventory_item)
        
        db.commit()
        print("✅ All data loaded successfully!")
        
        # Print summary
        print("\n📊 Data Summary:")
        print(f"- Distribution Centers: {df_centers.shape[0]} records")
        print(f"- Users: {df_users.shape[0]} records")
        print(f"- Products: {df_products.shape[0]} records")
        print(f"- Orders: {df_orders.shape[0]} records")
        print(f"- Order Items: {df_order_items.shape[0]} records")
        print(f"- Inventory Items: {df_inventory.shape[0]} records")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    load_csv_data()
