from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_, and_
from database import get_db, create_tables
from models import (
    User, Product, Order, OrderItem, InventoryItem, DistributionCenter,
    ConversationSession, ConversationMessage
)
import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import re
import logging
import json
from passlib.context import CryptContext
import jwt
from jwt.exceptions import InvalidTokenError
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create FastAPI app
app = FastAPI(
    title="E-commerce Conversational AI Chatbot API",
    description="Advanced conversational AI for E-commerce Customer Support with authentication",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()
    logger.info("Database tables created/verified")

# Enhanced Pydantic models for API
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional user context")

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime
    response_time_ms: Optional[int] = None
    data: Dict[str, Any] = {}
    suggestions: Optional[List[str]] = None

class UserInfo(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ConversationSummary(BaseModel):
    session_id: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message_preview: str

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception
    
    # For demo purposes, return a user object
    # In production, you'd query your user database
    return UserInfo(username=username)

# Enhanced database query functions
def get_top_selling_products(db: Session, limit: int = 5, category: Optional[str] = None) -> List[Dict]:
    """Get top selling products with optional category filter"""
    try:
        query = db.query(
            Product.name,
            Product.brand,
            Product.category,
            Product.retail_price,
            func.count(OrderItem.id).label('sales_count'),
            func.sum(OrderItem.sale_price).label('total_revenue')
        ).join(
            OrderItem, Product.id == OrderItem.product_id
        ).filter(
            OrderItem.status.in_(['Complete', 'Shipped', 'Processing'])
        )
        
        if category:
            query = query.filter(Product.category.ilike(f'%{category}%'))
        
        results = query.group_by(
            Product.id, Product.name, Product.brand, Product.category, Product.retail_price
        ).order_by(
            desc('sales_count')
        ).limit(limit).all()
        
        return [
            {
                "rank": i + 1,
                "product_name": result.name,
                "brand": result.brand,
                "category": result.category,
                "sales_count": result.sales_count,
                "total_revenue": float(result.total_revenue),
                "price": f"${result.retail_price:.2f}"
            }
            for i, result in enumerate(results)
        ]
    except Exception as e:
        logger.error(f"Error getting top products: {e}")
        return []

def get_order_details(db: Session, order_id: int) -> Optional[Dict]:
    """Get comprehensive order details"""
    try:
        order = db.query(Order).filter(Order.order_id == order_id).first()
        
        if not order:
            return None
        
        # Get user details
        user = db.query(User).filter(User.id == order.user_id).first()
        
        # Get order items with product details
        order_items = db.query(
            OrderItem, Product.name, Product.brand, Product.category
        ).join(
            Product, OrderItem.product_id == Product.id
        ).filter(
            OrderItem.order_id == order_id
        ).all()
        
        # Calculate order totals
        total_amount = sum(item.OrderItem.sale_price for item in order_items)
        
        return {
            "order_id": order.order_id,
            "status": order.status,
            "created_at": order.created_at,
            "shipped_at": order.shipped_at,
            "delivered_at": order.delivered_at,
            "returned_at": order.returned_at,
            "customer": {
                "name": f"{user.first_name} {user.last_name}" if user else "Unknown",
                "email": user.email if user else None,
                "city": user.city if user else None,
                "state": user.state if user else None
            },
            "items": [
                {
                    "product_name": item.Product.name,
                    "brand": item.Product.brand,
                    "category": item.Product.category,
                    "status": item.OrderItem.status,
                    "sale_price": float(item.OrderItem.sale_price),
                    "created_at": item.OrderItem.created_at,
                    "shipped_at": item.OrderItem.shipped_at,
                    "delivered_at": item.OrderItem.delivered_at
                }
                for item in order_items
            ],
            "summary": {
                "total_items": len(order_items),
                "total_amount": total_amount,
                "average_item_price": total_amount / len(order_items) if order_items else 0
            }
        }
    except Exception as e:
        logger.error(f"Error getting order details: {e}")
        return None

def search_products(db: Session, query: str, limit: int = 10) -> List[Dict]:
    """Advanced product search with multiple criteria"""
    try:
        search_terms = query.lower().split()
        
        # Build flexible search query
        search_conditions = []
        for term in search_terms:
            search_conditions.extend([
                Product.name.ilike(f'%{term}%'),
                Product.brand.ilike(f'%{term}%'),
                Product.category.ilike(f'%{term}%'),
                Product.department.ilike(f'%{term}%')
            ])
        
        products = db.query(Product).filter(
            or_(*search_conditions)
        ).limit(limit).all()
        
        # Get inventory count for each product
        results = []
        for product in products:
            stock_count = db.query(InventoryItem).filter(
                InventoryItem.product_id == product.id,
                InventoryItem.sold_at.is_(None)
            ).count()
            
            results.append({
                "id": product.id,
                "name": product.name,
                "brand": product.brand,
                "category": product.category,
                "department": product.department,
                "price": float(product.retail_price),
                "stock": stock_count,
                "sku": product.sku
            })
        
        return results
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return []

def get_user_order_history(db: Session, user_identifier: str, limit: int = 5) -> List[Dict]:
    """Get user's recent order history"""
    try:
        # Try to find user by email first, then by ID
        user = db.query(User).filter(
            or_(User.email == user_identifier, User.id == int(user_identifier) if user_identifier.isdigit() else 0)
        ).first()
        
        if not user:
            return []
        
        orders = db.query(Order).filter(
            Order.user_id == user.id
        ).order_by(
            desc(Order.created_at)
        ).limit(limit).all()
        
        order_history = []
        for order in orders:
            item_count = db.query(OrderItem).filter(OrderItem.order_id == order.order_id).count()
            order_history.append({
                "order_id": order.order_id,
                "status": order.status,
                "created_at": order.created_at,
                "item_count": item_count
            })
        
        return order_history
    except Exception as e:
        logger.error(f"Error getting user order history: {e}")
        return []

def process_advanced_query(message: str, db: Session, user_context: Optional[Dict] = None) -> tuple[str, Dict, List[str]]:
    """Advanced natural language query processing with suggestions"""
    message_lower = message.lower()
    suggestions = []
    
    # Product search queries
    if any(phrase in message_lower for phrase in ["find", "search", "show me", "looking for"]):
        # Extract search terms
        search_patterns = [
            r"find (.+?)(?:\?|$)",
            r"search (?:for )?(.+?)(?:\?|$)",
            r"show me (.+?)(?:\?|$)",
            r"looking for (.+?)(?:\?|$)"
        ]
        
        search_term = None
        for pattern in search_patterns:
            match = re.search(pattern, message_lower)
            if match:
                search_term = match.group(1).strip()
                break
        
        if search_term:
            products = search_products(db, search_term, limit=8)
            if products:
                response = f"**Found {len(products)} products matching '{search_term}':**\n\n"
                for product in products[:5]:  # Show top 5
                    stock_status = "In Stock" if product['stock'] > 0 else "Out of Stock"
                    response += f"• **{product['name']}** by {product['brand']}\n"
                    response += f"  - Category: {product['category']}\n"
                    response += f"  - Price: ${product['price']:.2f}\n"
                    response += f"  - Stock: {product['stock']} units ({stock_status})\n\n"
                
                if len(products) > 5:
                    response += f"*... and {len(products) - 5} more products*\n"
                
                suggestions = [
                    f"Show me more {search_term}",
                    f"What's the cheapest {search_term}?",
                    f"Which {search_term} is most popular?"
                ]
                
                return response, {"products": products, "search_term": search_term}, suggestions
            else:
                return f"No products found matching '{search_term}'. Try searching for categories like 'women', 'men', 'accessories', or specific brands.", {}, ["Show me all categories", "What brands do you have?", "Show me popular products"]
    
    # Enhanced top products query
    elif any(phrase in message_lower for phrase in ["top", "best", "popular", "most sold"]):
        limit = 5
        category = None
        
        # Extract number
        numbers = re.findall(r'\d+', message)
        if numbers:
            limit = min(int(numbers[0]), 20)
        
        # Extract category
        categories = ["women", "men", "accessories", "shoes", "swim", "clothing"]
        for cat in categories:
            if cat in message_lower:
                category = cat
                break
        
        products = get_top_selling_products(db, limit, category)
        if products:
            category_text = f" in {category}" if category else ""
            response = f"**Top {len(products)} best-selling products{category_text}:**\n\n"
            for product in products:
                response += f"{product['rank']}. **{product['product_name']}** by {product['brand']}\n"
                response += f"   • Sales: {product['sales_count']} units\n"
                response += f"   • Revenue: ${product['total_revenue']:,.2f}\n"
                response += f"   • Price: {product['price']}\n\n"
            
            suggestions = [
                "Show me inventory for these products",
                "What are the top products in women's category?",
                "Show me order details for popular products"
            ]
            
            return response, {"products": products, "category": category}, suggestions
        else:
            return "I couldn't find sales data at the moment.", {}, []
    
    # Enhanced order status query
    elif any(phrase in message_lower for phrase in ["order", "status", "track", "delivery"]):
        order_ids = re.findall(r'\b\d{4,}\b', message)
        if order_ids:
            order_id = int(order_ids[0])
            order_details = get_order_details(db, order_id)
            
            if order_details:
                response = f"**📦 Order #{order_details['order_id']} Details:**\n\n"
                response += f"**Status:** {order_details['status']}\n"
                response += f"**Customer:** {order_details['customer']['name']}\n"
                
                if order_details['customer']['email']:
                    response += f"**Email:** {order_details['customer']['email']}\n"
                
                response += f"**Order Date:** {order_details['created_at'].strftime('%Y-%m-%d %H:%M')}\n"
                
                if order_details['shipped_at']:
                    response += f"**Shipped:** {order_details['shipped_at'].strftime('%Y-%m-%d %H:%M')}\n"
                if order_details['delivered_at']:
                    response += f"**Delivered:** {order_details['delivered_at'].strftime('%Y-%m-%d %H:%M')}\n"
                
                response += f"\n**📊 Order Summary:**\n"
                response += f"• Total Items: {order_details['summary']['total_items']}\n"
                response += f"• Total Amount: ${order_details['summary']['total_amount']:.2f}\n"
                response += f"• Average Item Price: ${order_details['summary']['average_item_price']:.2f}\n"
                
                response += f"\n**📝 Items in Order:**\n"
                for item in order_details['items']:
                    item_status = f" ({item['status']})" if item['status'] != order_details['status'] else ""
                    response += f"• {item['product_name']} - ${item['sale_price']:.2f}{item_status}\n"
                
                suggestions = [
                    f"Show me more orders from {order_details['customer']['name']}",
                    "Track shipping status",
                    "Show me similar products"
                ]
                
                return response, {"order": order_details}, suggestions
            else:
                return f"Order #{order_id} not found. Please check the order number.", {}, ["Show me recent orders", "How do I track my order?"]
        else:
            return "Please provide an order ID (e.g., 'Check order 12345')", {}, ["Show me order format", "How do I find my order ID?"]
    
    # Enhanced inventory query
    elif any(phrase in message_lower for phrase in ["stock", "inventory", "available", "how many"]):
        # Extract product name
        product_patterns = [
            r"how many (.+?) (?:are|is|do|left)",
            r"(?:stock of|inventory of|availability of) (.+?)(?:\?|$)",
            r"(.+?) (?:in stock|available|inventory)",
            r"stock (?:for|of) (.+?)(?:\?|$)"
        ]
        
        product_name = None
        for pattern in product_patterns:
            match = re.search(pattern, message_lower)
            if match:
                product_name = match.group(1).strip()
                break
        
        if product_name:
            products = search_products(db, product_name, limit=5)
            if products:
                response = f"**📦 Inventory Status for '{product_name}':**\n\n"
                for product in products:
                    stock_status = "✅ In Stock" if product['stock'] > 10 else "⚠️ Low Stock" if product['stock'] > 0 else "❌ Out of Stock"
                    response += f"• **{product['name']}** by {product['brand']}\n"
                    response += f"  - Available: {product['stock']} units {stock_status}\n"
                    response += f"  - Price: ${product['price']:.2f}\n"
                    response += f"  - SKU: {product['sku']}\n\n"
                
                suggestions = [
                    "Show me alternative products",
                    "When will this be restocked?",
                    "Show me similar items in stock"
                ]
                
                return response, {"inventory": products}, suggestions
            else:
                return f"No products found matching '{product_name}'", {}, ["Show me all available products", "What categories do you have?"]
        else:
            # Show low stock alert
            low_stock_query = db.query(
                Product.name, Product.brand, 
                func.count(InventoryItem.id).label('stock_count')
            ).join(
                InventoryItem, Product.id == InventoryItem.product_id
            ).filter(
                InventoryItem.sold_at.is_(None)
            ).group_by(
                Product.id, Product.name, Product.brand
            ).having(
                func.count(InventoryItem.id) < 10
            ).order_by('stock_count').limit(10).all()
            
            if low_stock_query:
                response = "**⚠️ Low Stock Alert:**\n\n"
                for item in low_stock_query:
                    response += f"• {item.name} ({item.brand}): {item.stock_count} units left\n"
                
                return response, {"low_stock": [{"name": item.name, "brand": item.brand, "stock": item.stock_count} for item in low_stock_query]}, ["Show me restocking schedule", "Alert me when restocked"]
            else:
                return "All products are well stocked!", {}, ["Show me inventory by category"]
    
    # User history query (if authenticated)
    elif any(phrase in message_lower for phrase in ["my orders", "order history", "past orders"]):
        if user_context and user_context.get("user_id"):
            orders = get_user_order_history(db, str(user_context["user_id"]), limit=10)
            if orders:
                response = "**📋 Your Recent Orders:**\n\n"
                for order in orders:
                    response += f"• Order #{order['order_id']} - {order['status']}\n"
                    response += f"  Date: {order['created_at'].strftime('%Y-%m-%d')}\n"
                    response += f"  Items: {order['item_count']}\n\n"
                
                return response, {"orders": orders}, ["Show order details", "Track my latest order"]
            else:
                return "No order history found.", {}, ["Browse products", "How to place an order?"]
        else:
            return "Please log in to view your order history.", {}, ["How do I log in?", "Create an account"]
    
    # Default with helpful suggestions
    else:
        return """🤖 **I can help you with:**

• **Product Search** - "Find women's dresses" or "Search for Nike shoes"
• **Top Products** - "What are the top 5 most sold products?"
• **Order Tracking** - "Check order status 12345"
• **Inventory Check** - "How many Classic T-Shirts are in stock?"
• **Order History** - "Show my recent orders" (requires login)

**What would you like to know?**""", {}, [
            "Show me popular products",
            "Find women's accessories", 
            "Check order 12345",
            "What's in stock?",
            "Show me all categories"
        ]

# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access token"""
    # Demo authentication - in production, verify against database
    if form_data.username == "demo" and form_data.password == "password":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, 
            expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/users/me", response_model=UserInfo)
async def read_users_me(current_user: UserInfo = Depends(get_current_user)):
    """Get current user information"""
    return current_user

# Enhanced chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_message: ChatMessage,
    db: Session = Depends(get_db),
    current_user: Optional[UserInfo] = Depends(get_current_user)
):
    """Enhanced chat endpoint with authentication and advanced processing"""
    start_time = datetime.now()
    
    try:
        # Generate or use existing conversation ID
        conversation_id = chat_message.conversation_id or str(uuid.uuid4())
        
        # Determine user identifier
        user_identifier = current_user.username if current_user else "anonymous"
        
        # Create or get conversation session
        session = db.query(ConversationSession).filter(
            ConversationSession.session_id == conversation_id
        ).first()
        
        if not session:
            session = ConversationSession(
                session_id=conversation_id,
                user_identifier=user_identifier,
                is_active=True,
                session_metadata={"authenticated": bool(current_user)}
            )
            db.add(session)
            db.commit()
        
        # Save user message
        user_message = ConversationMessage(
            session_id=conversation_id,
            role="user",
            content=chat_message.message,
            message_type="chat",
            timestamp=datetime.now()
        )
        db.add(user_message)
        db.flush()
        
        # Process query with user context
        user_context = chat_message.user_context or {}
        if current_user:
            user_context["authenticated"] = True
            user_context["username"] = current_user.username
        
        response_text, response_data, suggestions = process_advanced_query(
            chat_message.message, db, user_context
        )
        
        # Calculate response time
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Save assistant response
        assistant_message = ConversationMessage(
            session_id=conversation_id,
            role="assistant",
            content=response_text,
            message_type="chat",
            query_metadata=response_data if response_data else None,
            response_time_ms=response_time,
            timestamp=datetime.now()
        )
        db.add(assistant_message)
        db.commit()
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            data=response_data,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Protected endpoint example
@app.get("/api/conversations/my-history")
async def get_my_conversations(
    limit: int = 10,
    current_user: UserInfo = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's conversation history (requires authentication)"""
    try:
        sessions = db.query(ConversationSession).filter(
            ConversationSession.user_identifier == current_user.username
        ).order_by(
            ConversationSession.updated_at.desc()
        ).limit(limit).all()
        
        conversation_summaries = []
        for session in sessions:
            # Get last message preview
            last_message = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session.session_id
            ).order_by(
                ConversationMessage.timestamp.desc()
            ).first()
            
            message_count = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session.session_id
            ).count()
            
            conversation_summaries.append(ConversationSummary(
                session_id=session.session_id,
                message_count=message_count,
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_message_preview=last_message.content[:100] + "..." if last_message and len(last_message.content) > 100 else last_message.content if last_message else ""
            ))
        
        return {"conversations": conversation_summaries}
        
    except Exception as e:
        logger.error(f"Error getting user conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversations: {str(e)}")

# Backward compatibility and other endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_legacy(
    chat_message: ChatMessage,
    db: Session = Depends(get_db)
):
    """Legacy chat endpoint for backward compatibility"""
    return await chat_endpoint(chat_message, db, None)

@app.get("/api/conversations/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get conversation history by ID"""
    try:
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == conversation_id
        ).order_by(ConversationMessage.timestamp).all()
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "response_time_ms": msg.response_time_ms
                }
                for msg in messages
            ]
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "ecommerce-chatbot",
        "version": "2.0.0",
        "timestamp": datetime.now()
    }

@app.get("/")
def read_root():
    return {
        "message": "E-commerce Conversational AI Chatbot API",
        "version": "2.0.0",
        "features": [
            "Advanced natural language processing",
            "User authentication with JWT",
            "Conversation history management",
            "Real-time product search",
            "Order tracking and status",
            "Inventory management",
            "Personalized responses"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
