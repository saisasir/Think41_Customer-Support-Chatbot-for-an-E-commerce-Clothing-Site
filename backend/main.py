from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="E-commerce Customer Support Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str

# Response model
class ChatResponse(BaseModel):
    response: str
    data: Dict[str, Any] = {}

# Data storage
class DataStore:
    def __init__(self):
        self.users = None
        self.products = None
        self.orders = None
        self.order_items = None
        self.inventory_items = None
        self.distribution_centers = None
        
    def load_data(self):
        """Load all CSV files into memory"""
        try:
            data_path = "/app/data"
            self.users = pd.read_csv(f"{data_path}/users.csv")
            self.products = pd.read_csv(f"{data_path}/products.csv")
            self.orders = pd.read_csv(f"{data_path}/orders.csv")
            self.order_items = pd.read_csv(f"{data_path}/order_items.csv")
            self.inventory_items = pd.read_csv(f"{data_path}/inventory_items.csv")
            self.distribution_centers = pd.read_csv(f"{data_path}/distribution_centers.csv")
            logger.info("All data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

# Initialize data store
data_store = DataStore()

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    data_store.load_data()

@app.get("/")
def read_root():
    return {"message": "E-commerce Customer Support Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat messages and return appropriate responses"""
    message = request.message.lower()
    
    try:
        # Check for top selling products query
        if "top" in message and ("sold" in message or "selling" in message):
            return handle_top_products_query(message)
        
        # Check for order status query
        elif "order" in message and ("status" in message or "show" in message):
            return handle_order_status_query(message)
        
        # Check for inventory query
        elif "stock" in message or "inventory" in message or "left" in message:
            return handle_inventory_query(message)
        
        # Check for product search
        elif "product" in message or "find" in message or "search" in message:
            return handle_product_search(message)
        
        # Check for user-related queries
        elif "customer" in message or "user" in message:
            return handle_user_query(message)
        
        else:
            return ChatResponse(
                response="I can help you with:\n" +
                        "1. Finding top selling products (e.g., 'What are the top 5 most sold products?')\n" +
                        "2. Checking order status (e.g., 'Show me the status of order ID 12345')\n" +
                        "3. Checking inventory (e.g., 'How many Classic T-Shirts are left in stock?')\n" +
                        "4. Searching for products (e.g., 'Find all women's accessories')\n" +
                        "5. Customer information (e.g., 'How many customers from California?')"
            )
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return ChatResponse(response=f"Sorry, I encountered an error: {str(e)}")

def handle_top_products_query(message: str) -> ChatResponse:
    """Handle queries about top selling products"""
    # Extract number from message (default to 5)
    import re
    numbers = re.findall(r'\d+', message)
    top_n = int(numbers[0]) if numbers else 5
    
    # Calculate top selling products
    sold_items = data_store.order_items[data_store.order_items['status'].isin(['Complete', 'Shipped', 'Processing'])]
    product_sales = sold_items.groupby('product_id').size().reset_index(name='sales_count')
    
    # Merge with product details
    top_products = product_sales.merge(data_store.products, left_on='product_id', right_on='id')
    top_products = top_products.nlargest(top_n, 'sales_count')
    
    # Format response
    products_list = []
    for idx, row in top_products.iterrows():
        products_list.append({
            'rank': idx + 1,
            'name': row['name'],
            'brand': row['brand'],
            'category': row['category'],
            'sales_count': int(row['sales_count']),
            'price': f"${row['retail_price']:.2f}"
        })
    
    response_text = f"Here are the top {top_n} most sold products:\n\n"
    for product in products_list:
        response_text += f"{product['rank']}. {product['name']} ({product['brand']})\n"
        response_text += f"   Category: {product['category']}\n"
        response_text += f"   Sales: {product['sales_count']} units\n"
        response_text += f"   Price: {product['price']}\n\n"
    
    return ChatResponse(
        response=response_text,
        data={'top_products': products_list}
    )

def handle_order_status_query(message: str) -> ChatResponse:
    """Handle queries about order status"""
    import re
    # Extract order ID from message
    order_id_match = re.search(r'\b(\d+)\b', message)
    
    if not order_id_match:
        return ChatResponse(response="Please provide an order ID. For example: 'Show me the status of order ID 12345'")
    
    order_id = int(order_id_match.group(1))
    
    # Find order
    order = data_store.orders[data_store.orders['order_id'] == order_id]
    
    if order.empty:
        return ChatResponse(response=f"Order ID {order_id} not found.")
    
    order_info = order.iloc[0]
    order_items = data_store.order_items[data_store.order_items['order_id'] == order_id]
    
    # Get order details
    response_text = f"Order ID: {order_id}\n"
    response_text += f"Status: {order_info['status']}\n"
    response_text += f"Customer ID: {order_info['user_id']}\n"
    response_text += f"Created: {order_info['created_at']}\n"
    response_text += f"Number of items: {order_info['num_of_item']}\n"
    
    if pd.notna(order_info.get('shipped_at')):
        response_text += f"Shipped: {order_info['shipped_at']}\n"
    if pd.notna(order_info.get('delivered_at')):
        response_text += f"Delivered: {order_info['delivered_at']}\n"
    if pd.notna(order_info.get('returned_at')):
        response_text += f"Returned: {order_info['returned_at']}\n"
    
    # Add item details
    if not order_items.empty:
        response_text += "\nOrder Items:\n"
        for idx, item in order_items.iterrows():
            product = data_store.products[data_store.products['id'] == item['product_id']]
            if not product.empty:
                product_info = product.iloc[0]
                response_text += f"- {product_info['name']} ({product_info['brand']}) - Status: {item['status']}\n"
    
    order_data = {
        'order_id': int(order_id),
        'status': order_info['status'],
        'user_id': int(order_info['user_id']),
        'created_at': order_info['created_at'],
        'num_items': int(order_info['num_of_item'])
    }
    
    return ChatResponse(response=response_text, data={'order': order_data})

def handle_inventory_query(message: str) -> ChatResponse:
    """Handle queries about inventory/stock levels"""
    # Extract product name from message
    product_keywords = []
    
    # Common product keywords to search for
    common_products = ['t-shirt', 'tshirt', 'shirt', 'jeans', 'dress', 'jacket', 'shoes', 'accessory', 'accessories']
    
    for keyword in common_products:
        if keyword in message.lower():
            product_keywords.append(keyword)
    
    if not product_keywords:
        # If no specific product mentioned, show low stock items
        return show_low_stock_items()
    
    # Search for products matching keywords
    matching_products = data_store.products[
        data_store.products['name'].str.lower().str.contains('|'.join(product_keywords), na=False)
    ]
    
    if matching_products.empty:
        return ChatResponse(response=f"No products found matching '{', '.join(product_keywords)}'")
    
    # Calculate inventory for matching products
    inventory_summary = []
    response_text = "Inventory Status:\n\n"
    
    for idx, product in matching_products.head(10).iterrows():
        # Count available inventory (created but not sold)
        available = data_store.inventory_items[
            (data_store.inventory_items['product_id'] == product['id']) & 
            (data_store.inventory_items['sold_at'].isna())
        ].shape[0]
        
        inventory_summary.append({
            'product_name': product['name'],
            'brand': product['brand'],
            'sku': product['sku'],
            'available_stock': available
        })
        
        response_text += f"• {product['name']} ({product['brand']})\n"
        response_text += f"  SKU: {product['sku']}\n"
        response_text += f"  Available Stock: {available} units\n\n"
    
    return ChatResponse(response=response_text, data={'inventory': inventory_summary})

def show_low_stock_items() -> ChatResponse:
    """Show items with low stock levels"""
    # Calculate stock levels for all products
    stock_levels = []
    
    for idx, product in data_store.products.iterrows():
        available = data_store.inventory_items[
            (data_store.inventory_items['product_id'] == product['id']) & 
            (data_store.inventory_items['sold_at'].isna())
        ].shape[0]
        
        if available < 50:  # Consider < 50 as low stock
            stock_levels.append({
                'product_name': product['name'],
                'brand': product['brand'],
                'available_stock': available
            })
    
    # Sort by stock level
    stock_levels.sort(key=lambda x: x['available_stock'])
    
    response_text = "Low Stock Alert (less than 50 units):\n\n"
    for item in stock_levels[:10]:  # Show top 10 low stock items
        response_text += f"• {item['product_name']} ({item['brand']}): {item['available_stock']} units\n"
    
    return ChatResponse(response=response_text, data={'low_stock_items': stock_levels[:10]})

def handle_product_search(message: str) -> ChatResponse:
    """Handle product search queries"""
    # Extract search criteria
    search_terms = []
    
    # Check for category
    categories = ['women', 'men', 'accessories', 'clothing', 'shoes', 'swim']
    for cat in categories:
        if cat in message.lower():
            search_terms.append(cat)
    
    if not search_terms:
        return ChatResponse(response="Please specify what type of products you're looking for (e.g., women's accessories, men's shoes)")
    
    # Search products
    query_str = '|'.join(search_terms)
    matching = data_store.products[
        (data_store.products['category'].str.lower().str.contains(query_str, na=False)) |
        (data_store.products['department'].str.lower().str.contains(query_str, na=False)) |
        (data_store.products['name'].str.lower().str.contains(query_str, na=False))
    ]
    
    if matching.empty:
        return ChatResponse(response=f"No products found matching '{', '.join(search_terms)}'")
    
    # Group by category
    category_counts = matching['category'].value_counts()
    
    response_text = f"Found {len(matching)} products matching your search:\n\n"
    for category, count in category_counts.head(10).items():
        response_text += f"• {category}: {count} products\n"
    
    # Show some example products
    response_text += "\nExample products:\n"
    for idx, product in matching.head(5).iterrows():
        response_text += f"- {product['name']} ({product['brand']}) - ${product['retail_price']:.2f}\n"
    
    return ChatResponse(
        response=response_text,
        data={'total_matches': len(matching), 'categories': category_counts.to_dict()}
    )

def handle_user_query(message: str) -> ChatResponse:
    """Handle queries about users/customers"""
    # Check for state-specific queries
    states = data_store.users['state'].unique()
    state_mentioned = None
    
    for state in states:
        if state.lower() in message.lower():
            state_mentioned = state
            break
    
    if state_mentioned:
        state_users = data_store.users[data_store.users['state'] == state_mentioned]
        return ChatResponse(
            response=f"There are {len(state_users)} customers from {state_mentioned}.\n" +
                    f"Average age: {state_users['age'].mean():.1f} years\n" +
                    f"Gender distribution: {state_users['gender'].value_counts().to_dict()}",
            data={'state': state_mentioned, 'customer_count': len(state_users)}
        )
    
    # General customer statistics
    total_users = len(data_store.users)
    avg_age = data_store.users['age'].mean()
    gender_dist = data_store.users['gender'].value_counts()
    top_states = data_store.users['state'].value_counts().head(5)
    
    response_text = f"Customer Statistics:\n\n"
    response_text += f"Total Customers: {total_users:,}\n"
    response_text += f"Average Age: {avg_age:.1f} years\n"
    response_text += f"\nGender Distribution:\n"
    for gender, count in gender_dist.items():
        response_text += f"• {gender}: {count:,} ({count/total_users*100:.1f}%)\n"
    response_text += f"\nTop 5 States:\n"
    for state, count in top_states.items():
        response_text += f"• {state}: {count:,} customers\n"
    
    return ChatResponse(response=response_text, data={'total_customers': total_users})

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}