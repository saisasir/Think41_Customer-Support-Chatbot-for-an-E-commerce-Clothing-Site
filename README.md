# Customer Support Chatbot for E-commerce Clothing Site

A full-stack customer support chatbot application for an e-commerce clothing site, built with FastAPI (backend) and vanilla JavaScript (frontend).

## Features

- **Top Selling Products**: Query the most popular products by sales volume
- **Order Status Tracking**: Check the status of specific orders by ID
- **Inventory Management**: View stock levels for products
- **Product Search**: Find products by category, department, or name
- **Customer Analytics**: View customer statistics and demographics

## Tech Stack

- **Backend**: Python, FastAPI, Pandas
- **Frontend**: HTML, CSS, JavaScript
- **Containerization**: Docker, Docker Compose
- **Web Server**: Nginx

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile          # Backend container configuration
├── frontend/
│   ├── index.html          # Chat interface
│   ├── nginx.conf          # Nginx configuration
│   └── Dockerfile          # Frontend container configuration
├── data/                   # CSV data files (place here)
│   ├── users.csv
│   ├── products.csv
│   ├── orders.csv
│   ├── order_items.csv
│   ├── inventory_items.csv
│   └── distribution_centers.csv
├── docker-compose.yml      # Container orchestration
└── README.md              # This file
```

## Prerequisites

- Docker and Docker Compose installed
- Git
- The e-commerce dataset CSV files

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/saisasir/Think41_Customer-Support-Chatbot-for-an-E-commerce-Clothing-Site.git
   cd Think41_Customer-Support-Chatbot-for-an-E-commerce-Clothing-Site
   ```

2. **Create the folder structure**
   ```bash
   mkdir -p backend frontend data
   ```

3. **Add the backend files**
   - Save `main.py` in the `backend/` directory
   - Save `requirements.txt` in the `backend/` directory
   - Save the backend `Dockerfile` in the `backend/` directory

4. **Add the frontend files**
   - Save `index.html` in the `frontend/` directory
   - Save `nginx.conf` in the `frontend/` directory
   - Save the frontend `Dockerfile` in the `frontend/` directory

5. **Add Docker Compose file**
   - Save `docker-compose.yml` in the root directory

6. **Add the dataset**
   - Download the e-commerce dataset
   - Place all CSV files in the `data/` directory

7. **Build and run the application**
   ```bash
   docker-compose up --build
   ```

8. **Access the application**
   - Frontend: http://localhost
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Usage Examples

### Via Chat Interface

1. **Top Selling Products**
   - "What are the top 5 most sold products?"
   - "Show me the top 10 best sellers"

2. **Order Status**
   - "Show me the status of order ID 12345"
   - "Check order 98765"

3. **Inventory Queries**
   - "How many Classic T-Shirts are left in stock?"
   - "Show me low stock items"
   - "Check inventory for jeans"

4. **Product Search**
   - "Find women's accessories"
   - "Search for men's shoes"
   - "Show me all swim products"

5. **Customer Analytics**
   - "How many customers from California?"
   - "Show customer statistics"

### Via API

You can also interact with the backend API directly:

```bash
# Chat endpoint
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What are the top 5 most sold products?"}'

# Health check
curl "http://localhost:8000/health"
```

## Development

### Running Backend Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Running Frontend Locally

Simply open `frontend/index.html` in a web browser, but update the API_URL in the script to point to `http://localhost:8000`.

## Docker Commands

```bash
# Build and start containers
docker-compose up --build

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Rebuild specific service
docker-compose build backend
docker-compose build frontend
```

## Troubleshooting

1. **Port conflicts**: If ports 80 or 8000 are already in use, modify the port mappings in `docker-compose.yml`

2. **Data not found**: Ensure all CSV files are placed in the `data/` directory before starting the containers

3. **CORS issues**: The backend is configured to allow all origins for development. For production, update the CORS settings in `main.py`

## Future Enhancements

- Add authentication and user sessions
- Implement real-time notifications for order updates
- Add multi-language support
- Integrate with payment systems
- Add voice input capabilities
- Implement recommendation engine
- Add export functionality for reports

## License

This project is for educational purposes as part of an interview assessment.