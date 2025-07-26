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
from groq import Groq
import asyncio
import time

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create FastAPI app
app = FastAPI(
    title="E-commerce Conversational AI Chatbot API",
    description="Advanced AI-powered conversational support for E-commerce with Groq LLM integration",
    version="3.0.0"
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
    
    # Test Groq connection
    try:
        test_response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        logger.info("✅ Groq LLM connection successful")
    except Exception as e:
        logger.error(f"❌ Groq LLM connection failed: {e}")

# Enhanced Pydantic models for API
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional user context")
    require_clarification: Optional[bool] = Field(False, description="Request clarification if needed")

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime
    response_time_ms: Optional[int] = None
    data: Dict[str, Any] = {}
    suggestions: Optional[List[str]] = None
    clarification_needed: Optional[bool] = False
    clarification_questions: Optional[List[str]] = None
    ai_confidence: Optional[float] = None
    query_type: Optional[str] = None

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
    ai_interactions: int

# Authentication functions (same as Milestone 4)
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
    
    return UserInfo(username=username)

# Enhanced database query functions with business intelligence
def get_comprehensive_product_analytics(db: Session, product_name: str = None) -> Dict:
    """Get comprehensive product analytics with AI insights"""
    try:
        base_query = db.query(
            Product.name,
            Product.brand,
            Product.category,
            Product.retail_price,
            func.count(OrderItem.id).label('total_sales'),
            func.sum(OrderItem.sale_price).label('total_revenue'),
            func.avg(OrderItem.sale_price).label('avg_sale_price'),
            func.count(func.distinct(Order.user_id)).label('unique_customers')
        ).join(
            OrderItem, Product.id == OrderItem.product_id
        ).join(
            Order, OrderItem.order_id == Order.order_id
        ).filter(
            OrderItem.status.in_(['Complete', 'Shipped', 'Processing'])
        )
        
        if product_name:
            base_query = base_query.filter(Product.name.ilike(f'%{product_name}%'))
        
        results = base_query.group_by(
            Product.id, Product.name, Product.brand, Product.category, Product.retail_price
        ).order_by(desc('total_sales')).limit(20).all()
        
        analytics = []
        for result in results:
            # Calculate business metrics
            profit_margin = ((result.avg_sale_price - result.retail_price) / result.retail_price * 100) if result.retail_price > 0 else 0
            
            analytics.append({
                "product_name": result.name,
                "brand": result.brand,
                "category": result.category,
                "metrics": {
                    "total_sales": result.total_sales,
                    "total_revenue": float(result.total_revenue),
                    "avg_sale_price": float(result.avg_sale_price),
                    "unique_customers": result.unique_customers,
                    "profit_margin_pct": round(profit_margin, 2),
                    "customer_retention": round(result.total_sales / result.unique_customers, 2)
                }
            })
        
        return {"analytics": analytics, "total_products": len(analytics)}
    except Exception as e:
        logger.error(f"Error in product analytics: {e}")
        return {"error": str(e)}

def get_customer_insights(db: Session, order_id: int = None, user_id: int = None) -> Dict:
    """Get comprehensive customer insights"""
    try:
        insights = {}
        
        if order_id:
            # Get order-specific customer insights
            order = db.query(Order).filter(Order.order_id == order_id).first()
            if order:
                user_id = order.user_id
        
        if user_id:
            # Get customer profile
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "Customer not found"}
            
            # Get customer's order history
            orders = db.query(Order).filter(Order.user_id == user_id).all()
            
            # Calculate customer metrics
            total_orders = len(orders)
            total_spent = db.query(func.sum(OrderItem.sale_price)).join(
                Order, OrderItem.order_id == Order.order_id
            ).filter(Order.user_id == user_id).scalar() or 0
            
            # Get favorite categories
            favorite_categories = db.query(
                Product.category,
                func.count(OrderItem.id).label('count')
            ).join(
                OrderItem, Product.id == OrderItem.product_id
            ).join(
                Order, OrderItem.order_id == Order.order_id
            ).filter(
                Order.user_id == user_id
            ).group_by(Product.category).order_by(desc('count')).limit(3).all()
            
            insights = {
                "customer_profile": {
                    "name": f"{user.first_name} {user.last_name}",
                    "email": user.email,
                    "location": f"{user.city}, {user.state}",
                    "age": user.age,
                    "member_since": user.created_at.strftime('%Y-%m-%d') if user.created_at else "Unknown"
                },
                "purchase_behavior": {
                    "total_orders": total_orders,
                    "total_spent": float(total_spent),
                    "average_order_value": float(total_spent / total_orders) if total_orders > 0 else 0,
                    "favorite_categories": [{"category": cat.category, "orders": cat.count} for cat in favorite_categories]
                }
            }
        
        return insights
    except Exception as e:
        logger.error(f"Error in customer insights: {e}")
        return {"error": str(e)}

def get_business_recommendations(db: Session, context: str) -> Dict:
    """Generate business recommendations based on data"""
    try:
        recommendations = []
        
        # Low stock alerts
        low_stock = db.query(
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
        ).order_by('stock_count').limit(5).all()
        
        if low_stock:
            recommendations.append({
                "type": "inventory_alert",
                "priority": "high",
                "message": f"Low stock alert: {len(low_stock)} products need restocking",
                "details": [{"product": f"{item.name} ({item.brand})", "stock": item.stock_count} for item in low_stock]
            })
        
        # Top performing products
        top_performers = db.query(
            Product.name,
            func.count(OrderItem.id).label('sales')
        ).join(
            OrderItem, Product.id == OrderItem.product_id
        ).filter(
            OrderItem.created_at >= datetime.now() - timedelta(days=30)
        ).group_by(Product.id, Product.name).order_by(desc('sales')).limit(3).all()
        
        if top_performers:
            recommendations.append({
                "type": "sales_insight",
                "priority": "medium",
                "message": "Top performing products this month",
                "details": [{"product": item.name, "sales": item.sales} for item in top_performers]
            })
        
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return {"error": str(e)}

def get_conversation_context(db: Session, conversation_id: str, limit: int = 10) -> List[Dict]:
    """Get recent conversation history for AI context"""
    try:
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == conversation_id
        ).order_by(
            ConversationMessage.timestamp.desc()
        ).limit(limit).all()
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "query_metadata": msg.query_metadata
            }
            for msg in reversed(messages)
        ]
    except Exception as e:
        logger.error(f"Error getting conversation context: {e}")
        return []

async def generate_ai_response(
    user_message: str, 
    conversation_history: List[Dict], 
    db_context: Dict,
    user_context: Optional[Dict] = None
) -> Dict:
    """Generate AI response using Groq LLM with business context"""
    try:
        # Build system prompt with business context
        system_prompt = f"""You are an expert AI assistant for an e-commerce clothing website. You have access to real-time data and should provide helpful, accurate responses.

BUSINESS CONTEXT:
- You help customers with product inquiries, order tracking, inventory checks, and general support
- Always be professional, friendly, and solution-oriented
- If you need more information, ask clarifying questions
- Use the provided data to give specific, actionable answers

AVAILABLE DATA CONTEXT:
{json.dumps(db_context, indent=2) if db_context else "No specific data context provided"}

CONVERSATION STYLE:
- Be conversational but professional
- Use emojis sparingly and appropriately
- Provide specific numbers and details when available
- Offer next steps or related suggestions
- If uncertain, be honest and ask for clarification

Remember: You have access to real customer data, orders, products, and inventory. Use this information to provide personalized, helpful responses."""

        # Prepare conversation messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 8 messages for context)
        for msg in conversation_history[-8:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current user message
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        # Generate AI response
        start_time = time.time()
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.9
        )
        
        response_time = int((time.time() - start_time) * 1000)
        ai_response = completion.choices[0].message.content
        
        # Calculate confidence based on response characteristics
        confidence = calculate_ai_confidence(ai_response, db_context)
        
        # Determine if clarification is needed
        needs_clarification = check_clarification_needed(ai_response, user_message)
        clarification_questions = generate_clarification_questions(user_message, ai_response) if needs_clarification else None
        
        # Generate suggestions
        suggestions = generate_contextual_suggestions(user_message, ai_response, db_context)
        
        return {
            "response": ai_response,
            "ai_confidence": confidence,
            "response_time_ms": response_time,
            "clarification_needed": needs_clarification,
            "clarification_questions": clarification_questions,
            "suggestions": suggestions,
            "tokens_used": completion.usage.total_tokens if hasattr(completion, 'usage') else None
        }
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. Please try rephrasing your question or contact support.",
            "ai_confidence": 0.0,
            "error": str(e)
        }

def calculate_ai_confidence(response: str, context: Dict) -> float:
    """Calculate confidence score for AI response"""
    confidence = 0.8  # Base confidence
    
    # Increase confidence if response contains specific data
    if any(keyword in response.lower() for keyword in ["order", "product", "stock", "price", "$"]):
        confidence += 0.1
    
    # Decrease confidence if response is vague
    if any(phrase in response.lower() for phrase in ["i'm not sure", "maybe", "perhaps", "it depends"]):
        confidence -= 0.2
    
    # Increase confidence if we have relevant context data
    if context and len(context) > 0:
        confidence += 0.1
    
    return max(0.0, min(1.0, confidence))

def check_clarification_needed(response: str, user_message: str) -> bool:
    """Check if the AI response indicates clarification is needed"""
    clarification_indicators = [
        "need more information",
        "could you clarify",
        "which specific",
        "please specify",
        "not sure which",
        "multiple options"
    ]
    
    return any(indicator in response.lower() for indicator in clarification_indicators)

def generate_clarification_questions(user_message: str, ai_response: str) -> List[str]:
    """Generate clarifying questions based on the context"""
    questions = []
    
    message_lower = user_message.lower()
    
    if "product" in message_lower and "find" in message_lower:
        questions.extend([
            "What specific category are you looking for? (e.g., women's, men's, accessories)",
            "Do you have a preferred brand in mind?",
            "What's your budget range?"
        ])
    
    if "order" in message_lower:
        questions.extend([
            "Do you have your order number handy?",
            "What's the email address associated with your order?"
        ])
    
    if "stock" in message_lower or "inventory" in message_lower:
        questions.extend([
            "What's the exact product name you're looking for?",
            "Do you need a specific size or color?"
        ])
    
    return questions[:3]  # Limit to 3 questions

def generate_contextual_suggestions(user_message: str, ai_response: str, context: Dict) -> List[str]:
    """Generate contextual suggestions based on conversation"""
    suggestions = []
    
    message_lower = user_message.lower()
    
    if "product" in message_lower or "find" in message_lower:
        suggestions.extend([
            "Show me similar products",
            "What are the most popular items?",
            "Check availability in my size"
        ])
    
    if "order" in message_lower:
        suggestions.extend([
            "Track my other orders",
            "View order history",
            "Contact customer service"
        ])
    
    if "top" in message_lower or "best" in message_lower:
        suggestions.extend([
            "Show me products in a specific category",
            "What's on sale this week?",
            "Find products under $50"
        ])
    
    # Add business-specific suggestions
    if context and "analytics" in context:
        suggestions.append("Show me detailed product analytics")
    
    return suggestions[:4]  # Limit to 4 suggestions

def classify_query_type(message: str) -> str:
    """Classify the type of user query for analytics"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["order", "track", "status", "delivery"]):
        return "order_inquiry"
    elif any(word in message_lower for word in ["product", "find", "search", "show"]):
        return "product_search"
    elif any(word in message_lower for word in ["stock", "inventory", "available"]):
        return "inventory_check"
    elif any(word in message_lower for word in ["top", "best", "popular", "most"]):
        return "analytics_query"
    elif any(word in message_lower for word in ["help", "support", "how", "what"]):
        return "general_support"
    else:
        return "general_conversation"

async def process_intelligent_query(
    message: str, 
    db: Session, 
    conversation_history: List[Dict],
    user_context: Optional[Dict] = None,
    require_clarification: bool = False
) -> tuple[str, Dict, List[str], bool, List[str], float, str]:
    """Process query with combined rule-based and AI intelligence"""
    
    query_type = classify_query_type(message)
    message_lower = message.lower()
    
    # Try rule-based processing first for specific queries
    db_context = {}
    rule_based_response = None
    
    # Product analytics queries
    if any(phrase in message_lower for phrase in ["analytics", "performance", "metrics", "insights"]):
        product_name = None
        # Extract product name if specified
        product_match = re.search(r"(?:for|of|about) (.+?)(?:\?|$)", message_lower)
        if product_match:
            product_name = product_match.group(1).strip()
        
        analytics = get_comprehensive_product_analytics(db, product_name)
        db_context["analytics"] = analytics
        
        if not analytics.get("error"):
            rule_based_response = f"**📊 Product Analytics {'for ' + product_name if product_name else ''}:**\n\n"
            for item in analytics["analytics"][:5]:
                rule_based_response += f"**{item['product_name']}** by {item['brand']}\n"
                rule_based_response += f"• Sales: {item['metrics']['total_sales']} units\n"
                rule_based_response += f"• Revenue: ${item['metrics']['total_revenue']:,.2f}\n"
                rule_based_response += f"• Avg Price: ${item['metrics']['avg_sale_price']:.2f}\n"
                rule_based_response += f"• Customers: {item['metrics']['unique_customers']}\n\n"
    
    # Order insights
    elif "order" in message_lower and ("insight" in message_lower or "customer" in message_lower):
        order_ids = re.findall(r'\b\d{4,}\b', message)
        if order_ids:
            order_id = int(order_ids[0])
            insights = get_customer_insights(db, order_id=order_id)
            db_context["customer_insights"] = insights
            
            if not insights.get("error"):
                customer = insights["customer_profile"]
                behavior = insights["purchase_behavior"]
                
                rule_based_response = f"**👤 Customer Insights for Order #{order_id}:**\n\n"
                rule_based_response += f"**Customer:** {customer['name']}\n"
                rule_based_response += f"**Location:** {customer['location']}\n"
                rule_based_response += f"**Member Since:** {customer['member_since']}\n\n"
                rule_based_response += f"**Purchase Behavior:**\n"
                rule_based_response += f"• Total Orders: {behavior['total_orders']}\n"
                rule_based_response += f"• Total Spent: ${behavior['total_spent']:.2f}\n"
                rule_based_response += f"• Avg Order Value: ${behavior['average_order_value']:.2f}\n"
                rule_based_response += f"• Favorite Categories: {', '.join([cat['category'] for cat in behavior['favorite_categories']])}\n"
    
    # Business recommendations
    elif "recommend" in message_lower or "suggest" in message_lower:
        recommendations = get_business_recommendations(db, message)
        db_context["recommendations"] = recommendations
        
        if recommendations.get("recommendations"):
            rule_based_response = "**💡 Business Recommendations:**\n\n"
            for rec in recommendations["recommendations"]:
                priority_emoji = "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🟢"
                rule_based_response += f"{priority_emoji} **{rec['message']}**\n"
                if rec.get("details"):
                    for detail in rec["details"][:3]:
                        if isinstance(detail, dict):
                            rule_based_response += f"  • {detail.get('product', detail.get('message', str(detail)))}\n"
                rule_based_response += "\n"
    
    # If no rule-based response or AI is preferred, use AI
    if not rule_based_response or require_clarification:
        ai_result = await generate_ai_response(message, conversation_history, db_context, user_context)
        
        return (
            ai_result.get("response", "I'm sorry, I couldn't process your request."),
            db_context,
            ai_result.get("suggestions", []),
            ai_result.get("clarification_needed", False),
            ai_result.get("clarification_questions", []),
            ai_result.get("ai_confidence", 0.5),
            query_type
        )
    else:
        # Use rule-based response but enhance with AI suggestions
        suggestions = generate_contextual_suggestions(message, rule_based_response, db_context)
        
        return (
            rule_based_response,
            db_context,
            suggestions,
            False,
            [],
            0.9,  # High confidence for rule-based responses
            query_type
        )

# Authentication endpoints (same as Milestone 4)
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
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
    return current_user

# Ultimate chat endpoint with full AI integration
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_message: ChatMessage,
    db: Session = Depends(get_db),
    current_user: Optional[UserInfo] = Depends(get_current_user)
):
    """Ultimate AI-powered chat endpoint with Groq LLM integration"""
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
                session_metadata={
                    "authenticated": bool(current_user),
                    "ai_enabled": True,
                    "version": "3.0.0"
                }
            )
            db.add(session)
            db.commit()
        
        # Get conversation history for context
        conversation_history = get_conversation_context(db, conversation_id)
        
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
        
        # Process query with AI intelligence
        user_context = chat_message.user_context or {}
        if current_user:
            user_context.update({
                "authenticated": True,
                "username": current_user.username
            })
        
        (
            response_text, 
            response_data, 
            suggestions,
            clarification_needed,
            clarification_questions,
            ai_confidence,
            query_type
        ) = await process_intelligent_query(
            chat_message.message,
            db,
            conversation_history,
            user_context,
            chat_message.require_clarification
        )
        
        # Calculate total response time
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Save assistant response with AI metadata
        assistant_message = ConversationMessage(
            session_id=conversation_id,
            role="assistant",
            content=response_text,
            message_type="ai_chat",
            query_metadata={
                "response_data": response_data,
                "ai_confidence": ai_confidence,
                "query_type": query_type,
                "clarification_needed": clarification_needed
            },
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
            suggestions=suggestions,
            clarification_needed=clarification_needed,
            clarification_questions=clarification_questions,
            ai_confidence=ai_confidence,
            query_type=query_type
        )
        
    except Exception as e:
        logger.error(f"Error in AI chat endpoint: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Analytics endpoint for monitoring AI performance
@app.get("/api/analytics/ai-performance")
async def get_ai_performance_analytics(
    days: int = 7,
    current_user: UserInfo = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI performance analytics (requires authentication)"""
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        # Get AI message statistics
        ai_messages = db.query(ConversationMessage).filter(
            ConversationMessage.message_type == "ai_chat",
            ConversationMessage.timestamp >= since_date
        ).all()
        
        if not ai_messages:
            return {"message": "No AI interactions found in the specified period"}
        
        # Calculate metrics
        total_interactions = len(ai_messages)
        avg_response_time = sum(msg.response_time_ms for msg in ai_messages if msg.response_time_ms) / len([msg for msg in ai_messages if msg.response_time_ms])
        
        # Confidence distribution
        confidences = [msg.query_metadata.get("ai_confidence", 0) for msg in ai_messages if msg.query_metadata]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Query types
        query_types = {}
        for msg in ai_messages:
            if msg.query_metadata and "query_type" in msg.query_metadata:
                query_type = msg.query_metadata["query_type"]
                query_types[query_type] = query_types.get(query_type, 0) + 1
        
        return {
            "period_days": days,
            "total_ai_interactions": total_interactions,
            "average_response_time_ms": round(avg_response_time, 2),
            "average_confidence": round(avg_confidence, 3),
            "query_type_distribution": query_types,
            "performance_score": round((avg_confidence * 0.7 + (1 - min(avg_response_time / 2000, 1)) * 0.3), 3)
        }
        
    except Exception as e:
        logger.error(f"Error getting AI analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

# Health check with AI status
@app.get("/health")
async def health_check():
    """Enhanced health check with AI status"""
    ai_status = "unknown"
    try:
        test_completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        ai_status = "healthy"
    except Exception as e:
        ai_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "ecommerce-chatbot",
        "version": "3.0.0",
        "ai_llm_status": ai_status,
        "features": [
            "Groq LLM Integration",
            "Advanced Business Intelligence", 
            "Context-Aware Conversations",
            "Clarifying Questions",
            "Performance Analytics"
        ],
        "timestamp": datetime.now()
    }

# Backward compatibility endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_legacy(
    chat_message: ChatMessage,
    db: Session = Depends(get_db)
):
    """Legacy chat endpoint with AI integration"""
    return await chat_endpoint(chat_message, db, None)

@app.get("/api/conversations/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get conversation history with AI interaction tracking"""
    try:
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == conversation_id
        ).order_by(ConversationMessage.timestamp).all()
        
        ai_interactions = len([msg for msg in messages if msg.message_type == "ai_chat"])
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "ai_interactions": ai_interactions,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "message_type": msg.message_type,
                    "response_time_ms": msg.response_time_ms,
                    "ai_metadata": msg.query_metadata if msg.message_type == "ai_chat" else None
                }
                for msg in messages
            ]
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.get("/")
def read_root():
    return {
        "message": "🤖 E-commerce Conversational AI Chatbot API",
        "version": "3.0.0",
        "powered_by": "Groq LLM + Advanced Business Intelligence",
        "capabilities": [
            "🧠 Natural Language Understanding with Groq LLM",
            "📊 Real-time Business Analytics & Insights", 
            "🔍 Intelligent Product Search & Recommendations",
            "📦 Comprehensive Order Tracking & Customer Insights",
            "💬 Context-Aware Conversations with Memory",
            "❓ Smart Clarifying Questions",
            "🔐 JWT Authentication & User Management",
            "📈 AI Performance Monitoring & Analytics"
        ],
        "ai_models": ["llama3-8b-8192"],
        "business_intelligence": True,
        "real_time_data": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
