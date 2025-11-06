# Fraud_detection using GE-GNN

# Aegis - Fraud Detection Platform

A full-stack web application for detecting fraudulent reviews using machine learning. The platform allows users to upload CSV files containing reviews, classify them using an AI model hosted on Google Colab, and view detailed analytics.

---

📁 Project Structure

```
aegis/
├── backend/                      # Flask backend server
│   ├── mongo_connection.py       # Main Flask app with all API endpoints
│   ├── admin_approve.py          # CLI tool to approve/deny access requests
│   ├── inspect_user.py           # CLI tool to check user status in database
│   ├── test_mongo.py             # MongoDB connection testing utility
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # Environment variables (MongoDB, API URLs)
│   └── ADMIN_README.md           # Admin API documentation
│
├── frontend/                     # React frontend application
│   ├── public/
│   │   └── index.html            # HTML template with page title
│   ├── src/
│   │   ├── components/           # Reusable React components
│   │   │   ├── CSVUpload.js      # CSV file upload and parsing component
│   │   │   ├── FraudChart.js     # Pie chart for fraud statistics
│   │   │   ├── Navbar.js         # Navigation bar with user menu
│   │   │   ├── NetworkGraph.js   # Network visualization component
│   │   │   ├── ProtectedRoute.js # Route wrapper for authentication
│   │   │   ├── ResultsTable.js   # Table displaying classification results
│   │   │   └── ui/               # Shadcn UI components (button, card, etc.)
│   │   ├── context/
│   │   │   └── AuthContext.js    # Authentication state management
│   │   ├── hooks/
│   │   │   └── use-toast.js      # Toast notification hook
│   │   ├── lib/
│   │   │   └── utils.js          # Utility functions (cn for classnames)
│   │   ├── mock/
│   │   │   └── mockData.js       # Mock data for testing and demos
│   │   ├── pages/                # Main application pages
│   │   │   ├── Dashboard.js      # Main dashboard with CSV upload and results
│   │   │   ├── Feedback.js       # User feedback submission page
│   │   │   ├── Login.js          # User login page
│   │   │   └── RequestAccess.js  # New user registration page
│   │   ├── utils/
│   │   │   └── csvParser.js      # CSV parsing utilities
│   │   ├── App.js                # Main app component with routing
│   │   ├── index.js              # React entry point
│   │   └── index.css             # Global styles with Tailwind
│   ├── package.json              # Node.js dependencies and scripts
│   └── tailwind.config.js        # Tailwind CSS configuration
│
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🔧 Backend Setup

### Prerequisites
- Python 3.8+
- MongoDB Atlas account (or local MongoDB)
- Google Colab notebook with fraud detection model running

### Installation Steps

1. **Navigate to backend directory:**
   ```powershell
   cd backend
   ```

2. **Create virtual environment (recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   
   Edit `backend/.env` file:
   ```env
   MONGO_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority
   MONGO_DB_NAME=fraud_detection
   PORT=5000
   COLAB_MODEL_URL=https://your-ngrok-url.ngrok-free.dev
   ```

   Replace:
   - `<username>` and `<password>` with your MongoDB credentials
   - `COLAB_MODEL_URL` with your ngrok tunnel URL from Colab

5. **Test MongoDB connection:**
   ```powershell
   python test_mongo.py
   ```

6. **Start the backend server:**
   ```powershell
   python mongo_connection.py
   ```

   Server will run on `http://localhost:5000`

### Backend API Endpoints

#### Authentication
- `POST /api/auth/login` - User login
- `POST /api/access-requests` - Request new account
- `GET /api/access-requests` - List access requests (admin)
- `POST /api/access-requests/<username>/approve` - Approve user
- `POST /api/access-requests/<username>/deny` - Deny user

#### Classification
- `POST /api/classify` - Classify reviews (proxies to Colab model)
- `POST /api/classifications` - Save classification results to database
- `GET /api/classifications` - Get classification history

#### Feedback
- `POST /api/feedback` - Submit platform feedback
- `GET /api/feedback` - Get feedback list

### Admin Tools

**Approve/Deny Users:**
```powershell
python admin_approve.py
```

**Check User Status:**
```powershell
python inspect_user.py <username>
```

---

## 🎨 Frontend Setup

### Prerequisites
- Node.js 16+ and npm

### Installation Steps

1. **Navigate to frontend directory:**
   ```powershell
   cd frontend
   ```

2. **Install dependencies:**
   ```powershell
   npm install
   ```

3. **Configure environment (optional):**
   
   Create `frontend/.env` file if backend is not on localhost:5000:
   ```env
   REACT_APP_BACKEND_URL=http://localhost:5000
   ```

4. **Start the development server:**
   ```powershell
   npm start
   ```

   Application will open at `http://localhost:3000`

5. **Build for production:**
   ```powershell
   npm run build
   ```

   Production files will be in `frontend/build/`

---

## 🚀 Running the Complete Application

### Step 1: Start MongoDB
Ensure MongoDB Atlas is accessible or local MongoDB is running.

### Step 2: Start Colab Model
1. Open your Google Colab notebook
2. Run the Flask server cell
3. Start ngrok tunnel
4. Copy the ngrok URL (e.g., `https://xxxxx.ngrok-free.dev`)
5. Update `COLAB_MODEL_URL` in `backend/.env`

### Step 3: Start Backend
```powershell
cd backend
python mongo_connection.py
```

### Step 4: Start Frontend
```powershell
cd frontend
npm start
```

### Step 5: Access Application
Open browser and go to `http://localhost:3000`

---

## 📊 Database Collections

The application uses the following MongoDB collections:

- **approved_users** - Approved user accounts (username, email, password, role)
- **access_requests** - Pending/denied access requests
- **classifications** - Saved classification results with predictions
- **platform_feedback** - User feedback submissions
- **model_feedback** - Model-specific feedback (optional)

---

## 🔐 Default Admin Setup

To create the first admin user, use the CLI tool:

```powershell
cd backend
python admin_approve.py
```

Or manually insert into MongoDB:
```javascript
db.approved_users.insertOne({
  username: "admin",
  email: "admin@aegis.com",
  password: "admin123",  // Change this!
  role: "admin",
  created_at: new Date()
})
```

---

## 🗑️ Unnecessary Files (Can be Removed)

### Backend
- `test_mongo.py` - Only needed for initial setup testing
- `inspect_user.py` - Only needed for debugging user issues
- `ADMIN_README.md` - Redundant with this README

### Frontend
- `CLEANUP_INSTRUCTIONS.md` - Temporary file, can be deleted

---

## 🛠️ Technology Stack

### Backend
- **Flask** - Python web framework
- **PyMongo** - MongoDB driver
- **Flask-CORS** - Cross-origin resource sharing
- **python-dotenv** - Environment variable management
- **certifi** - SSL certificate handling

### Frontend
- **React 18** - UI framework
- **React Router v6** - Client-side routing
- **Tailwind CSS** - Utility-first CSS framework
- **Shadcn/ui** - Component library
- **Radix UI** - Headless UI components
- **Recharts** - Chart library
- **Lucide React** - Icon library

### External Services
- **MongoDB Atlas** - Cloud database
- **Google Colab** - ML model hosting
- **ngrok** - Secure tunneling for Colab

---

## 📝 Usage Workflow

1. **Register Account:**
   - Go to `/request-access`
   - Fill in username, email, password
   - Wait for admin approval

2. **Admin Approves:**
   - Run `python admin_approve.py`
   - Select pending request
   - Approve user

3. **Login:**
   - Go to `/login`
   - Enter credentials
   - Access dashboard

4. **Classify Reviews:**
   - Upload CSV file with reviews
   - Click "Classify Reviews"
   - View results in table and chart
   - Download results as CSV

5. **Submit Feedback:**
   - Go to `/feedback`
   - Rate platform and provide comments
   - Submit feedback

---

## 🐛 Troubleshooting

### Backend Issues

**MongoDB Connection Failed:**
- Check `MONGO_URL` in `.env`
- Verify MongoDB Atlas IP whitelist (allow 0.0.0.0/0 for testing)
- Run `python test_mongo.py` to diagnose

**Colab Model Not Responding:**
- Ensure Colab notebook is running
- Check ngrok tunnel is active
- Verify `COLAB_MODEL_URL` in `.env`
- Check Colab logs for errors

**Port Already in Use:**
```powershell
# Find process using port 5000
netstat -ano | findstr :5000
# Kill the process
taskkill /PID <process_id> /F
```

### Frontend Issues

**Cannot Connect to Backend:**
- Ensure backend is running on port 5000
- Check `REACT_APP_BACKEND_URL` in `.env`
- Check browser console for CORS errors

**Build Errors:**
```powershell
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Port 3000 Already in Use:**
- React will prompt to use another port (e.g., 3001)
- Or kill the process using port 3000

---

## 📄 License

This project is for educational purposes.

---

## 👥 Support

For issues or questions:
1. Check the troubleshooting section
2. Review backend logs in terminal
3. Check browser console for frontend errors
4. Verify all services are running (MongoDB, Backend, Frontend, Colab)

---

**Last Updated:** November 2025
