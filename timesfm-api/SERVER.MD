 FastAPI Documentation:                                                                                                                                                                                                                       
  - ✅ Swagger UI: http://localhost:8000/docs - Interactive API documentation                                                                                                                                                                  
  - ✅ ReDoc: http://localhost:8000/redoc - Alternative API documentation                                                                                                                                                                      
  - ✅ OpenAPI Spec: http://localhost:8000/openapi.json - Machine-readable schema                                                                                                                                                              
                                                                                                                                                                                                                                               
  Static Documentation:                                                                                                                                                                                                                        
  - ✅ Production Runbook: /workspace/timesfm/timesfm-api/docs/production_runbook.md                                                                                                                                                           
  - ✅ Usage Examples: /workspace/timesfm/timesfm-api/examples/server_usage_example.sh                                                                                                                                                         
                                                                                                                                                                                                                                               
  🔧 Available API Endpoints                                                                                                                                                                                                                   
                                                                                                                                                                                                                                               
  /health                 # Server health check                                                                                                                                                                                                
  /forecast              # Time series forecasting                                                                                                                                                                                             
  /preprocess            # Data preprocessing with TimesFM 2.5 best practices                                                                                                                                                                  
  /preprocess/methods    # Available preprocessing methods
  /model/info            # Model information
  /stats                 # Usage statistics

  💻 Server Management Commands

  # Start server (background)
  ./scripts/start_server.sh start

  # Start with custom settings
  ./scripts/start_server.sh start PORT=8080 WORKERS=4

  # Check server status
  ./scripts/start_server.sh status

  # View logs
  ./scripts/start_server.sh logs
  ./scripts/start_server.sh logs -f  # Follow logs

  # Stop server
  ./scripts/start_server.sh stop

  # Restart server
  ./scripts/start_server.sh restart
