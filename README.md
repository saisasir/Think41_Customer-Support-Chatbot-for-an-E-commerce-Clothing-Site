# E-commerce Conversational AI Chatbot

A full-stack conversational AI chatbot for e-commerce customer support, built with FastAPI, PostgreSQL, and Groq LLM integration.

## Features

- **Database Integration**: PostgreSQL with comprehensive e-commerce data model
- **Conversational AI**: Groq LLM integration for natural language understanding
- **Data Queries**: Intelligent processing of customer queries about orders, products, and inventory
- **Conversation History**: Persistent chat sessions with context awareness
- **RESTful API**: Complete FastAPI backend with OpenAPI documentation
- **Containerized**: Docker Compose setup for easy deployment

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### Setup

1. **Clone the repository**
git clone <your-repository-url>
cd ecommerce-chatbot

2. **Set up environment variables**

Create root .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

Create backend .env file
echo "DATABASE_URL=postgresql://user:password@localhost:5432/ecommerce_chatbot" > backend/.env
echo "GROQ_API_KEY=your_groq_api_key_here" >> backend/.env


3. **Start the application**
docker-compose up --build


4. **Load sample data**
Wait for services to be ready, then:
docker-compose exec backend python load_data.py


5. **Access the application**
- Frontend: http://localhost
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Usage

### Chat Endpoint
curl -X POST http://localhost:8000/api/chat
-H "Content-Type: application/json"
-d '{"message": "What are the top 5 most sold products?"}'

### Example Queries
- "What are the top 5 most sold products?"
- "Show me the status of order ID 12345"
- "How many Classic T-Shirts are left in stock?"
- "What products does Nike sell?"

## Architecture

- **Backend**: FastAPI with SQLAlchemy ORM
- **Database**: PostgreSQL with conversation history
- **Frontend**: HTML/CSS/JavaScript with Nginx
- **AI**: Groq LLM integration for natural language processing
- **Deployment**: Docker Compose with health checks

## Development

### Local Development Setup
Start database only
docker-compose up db -d

Install dependencies
cd backend
pip install -r requirements.txt

Run backend locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000

### Database Schema
The application uses 6 main tables:
- `users`: Customer information
- `products`: Product catalog
- `orders`: Order records
- `order_items`: Order line items
- `inventory_items`: Product inventory
- `distribution_centers`: Warehouse locations

Plus conversation management:
- `conversation_sessions`: Chat sessions
- `conversation_messages`: Individual messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License