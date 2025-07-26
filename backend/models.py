from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Boolean
from sqlalchemy.relationship import relationship
from sqlalchemy.sql import func
from database import Base

# E-commerce Data Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    email = Column(String(255), unique=True, index=True)
    age = Column(Integer)
    gender = Column(String(10))
    state = Column(String(100))
    street_address = Column(String(255))
    postal_code = Column(String(20))
    city = Column(String(100))
    country = Column(String(100))
    latitude = Column(Float)
    longitude = Column(Float)
    traffic_source = Column(String(100))
    created_at = Column(DateTime)

class DistributionCenter(Base):
    __tablename__ = "distribution_centers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    cost = Column(Float)
    category = Column(String(100), index=True)
    name = Column(String(255), index=True)
    brand = Column(String(100), index=True)
    retail_price = Column(Float)
    department = Column(String(100), index=True)
    sku = Column(String(100), unique=True, index=True)
    distribution_center_id = Column(Integer, ForeignKey("distribution_centers.id"))

class Order(Base):
    __tablename__ = "orders"
    
    order_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    status = Column(String(50), index=True)
    gender = Column(String(10))
    created_at = Column(DateTime, index=True)
    returned_at = Column(DateTime, nullable=True)
    shipped_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    num_of_item = Column(Integer)

class OrderItem(Base):
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.order_id"), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    inventory_item_id = Column(Integer, ForeignKey("inventory_items.id"))
    status = Column(String(50), index=True)
    created_at = Column(DateTime, index=True)
    shipped_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    returned_at = Column(DateTime, nullable=True)
    sale_price = Column(Float)

class InventoryItem(Base):
    __tablename__ = "inventory_items"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    created_at = Column(DateTime, index=True)
    sold_at = Column(DateTime, nullable=True, index=True)
    cost = Column(Float)
    product_category = Column(String(100))
    product_name = Column(String(255))
    product_brand = Column(String(100))
    product_retail_price = Column(Float)
    product_department = Column(String(100))
    product_sku = Column(String(100))
    product_distribution_center_id = Column(Integer)

# Conversation Models (for future milestones)
class ConversationSession(Base):
    __tablename__ = "conversation_sessions"
    
    session_id = Column(String(36), primary_key=True, index=True)
    user_identifier = Column(String(255), index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True, index=True)
    session_metadata = Column(JSON, nullable=True)

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), ForeignKey("conversation_sessions.session_id"), index=True)
    role = Column(String(20), index=True)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.now(), index=True)
    message_type = Column(String(50), default="chat")
    query_metadata = Column(JSON, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
