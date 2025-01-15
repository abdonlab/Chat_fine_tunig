# Chat_fine_tunig
Chat con IAT5 entrenado CustomFit-V1
chatbot_project/
│
├── config/
│   ├── settings.py           # Configuraciones generales (API keys, DB, etc.)
│   └── constants.py          # Constantes globales
│
├── data/
│   ├── intents.json          # Archivo de entrenamiento (intenciones del bot)
│   ├── responses.json        # Respuestas predefinidas
│   └── user_data/            # Almacenamiento de datos de usuario (si es necesario)
│
├── models/
│   ├── chatbot_model.py      # Definición del modelo de IA
│   └── training_script.py    # Script para entrenar el modelo (si aplicable)
│
├── scripts/
│   ├── data_preprocessing.py # Preprocesamiento de datos
│   └── utils.py              # Funciones auxiliares
│
├── server/
│   ├── app.py                # Archivo principal de Flask/FastAPI
│   └── routes/
│       ├── chat_routes.py    # Endpoints relacionados con el chatbot
│       └── user_routes.py    # Endpoints relacionados con usuarios
│
├── static/
│   ├── css/                  # Archivos CSS
│   ├── js/                   # Archivos JavaScript
│   └── images/               # Imágenes
│
├── templates/
│   ├── index.html            # Interfaz de usuario (si usas HTML)
│   └── chat.html             # Página de chat
│
├── tests/
│   ├── test_chatbot.py       # Pruebas unitarias para el chatbot
│   └── test_routes.py        # Pruebas de las rutas del backend
│
├── logs/
│   └── chatbot.log           # Registro de actividades del bot
│
├── requirements.txt          # Dependencias del proyecto
│
├── .env                      # Variables de entorno (claves secretas, etc.)
│
└── README.md                 # Documentación del proyecto
