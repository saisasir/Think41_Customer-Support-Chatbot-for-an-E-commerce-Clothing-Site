from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from database import get_db, create_tables
from models import (
    User, Product, Order, OrderItem, InventoryItem, DistributionCenter,
    ConversationSession, ConversationMessage
)
import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="E-commerce Chatbot API",
    description="Conversational AI for E-commerce Customer Support",
    version="1.0.0"
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

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime
    data: Dict[str, Any] = {}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ecommerce-chatbot"}

@app.get("/")
def read_root():
    return {"message": "E-commerce Customer Support Chatbot API"}

# Enhanced database query functions
def get_top_selling_products(db: Session, limit: int = 5) -> List[Dict]:
    """Get top selling products from database"""
    try:
        results = db.query(
            Product.name,
            Product.brand,
            Product.category,
            Product.retail_price,
            func.count(OrderItem.id).label('sales_count')
        ).join(
            OrderItem, Product.id == OrderItem.product_id
        ).filter(
            OrderItem.status.in_(['Complete', 'Shipped', 'Processing'])
        ).group_by(
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
                "price": f"${result.retail_price:.2f}"
            }
            for i, result in enumerate(results)
        ]
    except Exception as e:
        logger.error(f"Error getting top products: {e}")
        return []

def get_order_status(db: Session, order_id: int) -> Optional[Dict]:
    """Get order status and details from database"""
    try:
        order = db.query(Order).filter(Order.order_id == order_id).first()
        
        if not order:
            return None
        
        # Get user details
        user = db.query(User).filter(User.id == order.user_id).first()
        
        # Get order items
        order_items = db.query(
            OrderItem, Product.name, Product.brand
        ).join(
            Product, OrderItem.product_id == Product.id
        ).filter(
            OrderItem.order_id == order_id
        ).all()
        
        return {
            "order_id": order.order_id,
            "status": order.status,
            "created_at": order.created_at,
            "shipped_at": order.shipped_at,
            "delivered_at": order.delivered_at,
            "customer_name": f"{user.first_name} {user.last_name}" if user else "Unknown",
            "customer_id": order.user_id,
            "num_items": order.num_of_item,
            "items": [
                {
                    "product_name": item.Product.name,
                    "brand": item.Product.brand,
                    "status": item.OrderItem.status,
                    "sale_price": item.OrderItem.sale_price
                }
                for item in order_items
            ]
        }
    except Exception as e:
        logger.error(f"Error getting order status: {e}")
        return None

def get_inventory_by_product_name(db: Session, product_name: str) -> Dict:
    """Get inventory count for a specific product from database"""
    try:
        # Find products matching the name (case-insensitive)
        products = db.query(Product).filter(
            Product.name.ilike(f'%{product_name}%')
        ).all()
        
        if not products:
            return {"error": f"No products found matching '{product_name}'"}
        
        inventory_info = []
        for product in products:
            available_count = db.query(InventoryItem).filter(
                InventoryItem.product_id == product.id,
                InventoryItem.sold_at.is_(None)
            ).count()
            
            inventory_info.append({
                "product_name": product.name,
                "brand": product.brand,
                "sku": product.sku,
                "available_stock": available_count,
                "retail_price": product.retail_price
            })
        
        return {"products": inventory_info}
    except Exception as e:
        logger.error(f"Error getting inventory: {e}")
        return {"error": f"Error checking inventory: {str(e)}"}

def process_natural_language_query(message: str, db: Session) -> tuple[str, Dict]:
    """Process natural language queries and return appropriate responses"""
    message_lower = message.lower()
    
    # Top selling products query
    if any(phrase in message_lower for phrase in ["top", "best selling", "most sold", "popular products"]):
        try:
            limit = 5
            # Extract number if specified
            numbers = re.findall(r'\d+', message)
            if numbers:
                limit = min(int(numbers[0]), 20)  # Cap at 20
            
            products = get_top_selling_products(db, limit)
            if products:
                response = f"Here are the top {len(products)} most sold products:\n\n"
                for product in products:
                    response += f"{product['rank']}. **{product['product_name']}** by {product['brand']}\n"
                    response += f"   Category: {product['category']}\n"
                    response += f"   Sales: {product['sales_count']} units | Price: {product['price']}\n\n"
                return response, {"top_products": products}
            else:
                return "I couldn't find any sales data at the moment.", {}
                
        except Exception as e:
            return f"I encountered an error while fetching the top products: {str(e)}", {}
    
    # Order status query
    elif any(phrase in message_lower for phrase in ["order", "status", "track"]):
        try:
            # Extract order ID
            order_ids = re.findall(r'\b\d{4,}\b', message)
            if order_ids:
                order_id = int(order_ids[0])
                order_info = get_order_status(db, order_id)
                
                if order_info:
                    response = f"**Order #{order_info['order_id']} Status:**\n\n"
                    response += f"• **Status:** {order_info['status']}\n"
                    response += f"• **Customer:** {order_info['customer_name']}\n"
                    response += f"• **Created:** {order_info['created_at'].strftime('%Y-%m-%d %H:%M') if order_info['created_at'] else 'N/A'}\n"
                    
                    if order_info['shipped_at']:
                        response += f"• **Shipped:** {order_info['shipped_at'].strftime('%Y-%m-%d %H:%M')}\n"
                    if order_info['delivered_at']:
                        response += f"• **Delivered:** {order_info['delivered_at'].strftime('%Y-%m-%d %H:%M')}\n"
                    
                    response += f"\n**Items in this order:**\n"
                    for item in order_info['items']:
                        response += f"- {item['product_name']} by {item['brand']} (${item['sale_price']})\n"
                    
                    return response, {"order": order_info}
                else:
                    return f"I couldn't find any order with ID #{order_id}. Please check the order number and try again.", {}
            else:
                return "Please provide an order ID number to check the status (e.g., 'What's the status of order 12345?')", {}
                
        except Exception as e:
            return f"I encountered an error while checking the order status: {str(e)}", {}
    
    # Inventory/stock query
    elif any(phrase in message_lower for phrase in ["stock", "inventory", "available", "how many"]):
        try:
            # Try to extract product name
            product_patterns = [
                r"how many (.+?) (?:are|is|do)",
                r"(?:stock of|inventory of) (.+?)(?:\?|$)",
                r"(.+?) (?:in stock|available)",
                r"stock (?:for|of) (.+?)(?:\?|$)"
            ]
            
            product_name = None
            for pattern in product_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    product_name = match.group(1).strip()
                    break
            
            if product_name:
                inventory_info = get_inventory_by_product_name(db, product_name)
                
                if "error" in inventory_info:
                    return inventory_info["error"], {}
                
                products = inventory_info["products"]
                if products:
                    response = f"**Inventory information for '{product_name}':**\n\n"
                    for product in products:
                        response += f"• **{product['product_name']}** by {product['brand']}\n"
                        response += f"  - Available stock: {product['available_stock']} units\n"
                        response += f"  - Price: ${product['retail_price']}\n"
                        response += f"  - SKU: {product['sku']}\n\n"
                    return response, {"inventory": products}
                else:
                    return f"No products found matching '{product_name}'", {}
            else:
                return "Please specify which product you'd like to check inventory for (e.g., 'How many Classic T-Shirts are in stock?')", {}
                
        except Exception as e:
            return f"I encountered an error while checking inventory: {str(e)}", {}
    
    # Default response
    else:
        return """I can help you with:

• **Top selling products** - Ask "What are the top 5 most sold products?"
• **Order status** - Ask "What's the status of order 12345?"
• **Inventory levels** - Ask "How many Classic T-Shirts are in stock?"

What would you like to know?""", {}

# Enhanced chat endpoint with conversation history
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_message: ChatMessage,
    db: Session = Depends(get_db)
):
    try:
        # Generate or use existing conversation ID
        conversation_id = chat_message.conversation_id or str(uuid.uuid4())
        
        # Create or get conversation session
        session = db.query(ConversationSession).filter(
            ConversationSession.session_id == conversation_id
        ).first()
        
        if not session:
            session = ConversationSession(
                session_id=conversation_id,
                user_identifier="anonymous",
                is_active=True
            )
            db.add(session)
            db.commit()
        
        # Save user message
        user_message = ConversationMessage(
            session_id=conversation_id,
            role="user",
            content=chat_message.message,
            message_type="chat"
        )
        db.add(user_message)
        
        # Process the query and generate response
        response_text, response_data = process_natural_language_query(chat_message.message, db)
        
        # Save assistant response
        assistant_message = ConversationMessage(
            session_id=conversation_id,
            role="assistant",
            content=response_text,
            message_type="chat",
            query_metadata=response_data if response_data else None
        )
        db.add(assistant_message)
        db.commit()
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Backward compatibility endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_legacy(
    chat_message: ChatMessage,
    db: Session = Depends(get_db)
):
    """Legacy chat endpoint for backward compatibility"""
    return await chat_endpoint(chat_message, db)

# Get conversation history
@app.get("/api/conversations/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    db: Session = Depends(get_db)
):
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
                    "message_type": msg.message_type
                }
                for msg in messages
            ]
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

# List all conversation sessions
@app.get("/api/conversations")
async def list_conversations(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    try:
        sessions = db.query(ConversationSession).order_by(
            ConversationSession.updated_at.desc()
        ).limit(limit).all()
        
        return {
            "conversations": [
                {
                    "session_id": session.session_id,
                    "user_identifier": session.user_identifier,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "is_active": session.is_active
                }
                for session in sessions
            ]
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")
