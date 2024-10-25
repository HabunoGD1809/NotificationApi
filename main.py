# Librerías estándar de Python
import asyncio
import os
import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from contextlib import contextmanager, asynccontextmanager
from functools import wraps

# Librerías de terceros
import jwt
import bcrypt
import uvicorn
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from colorama import init, Fore
from psycopg2 import IntegrityError

# FastAPI
from fastapi import Body, FastAPI, Depends, HTTPException, Query, Request, status, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState

# SQLAlchemy (ORM para bases de datos)
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, ForeignKey, Text, func, case
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, Session, joinedload, relationship

# Pydantic (para validación de datos en FastAPI)
from pydantic import BaseModel, ValidationError, field_validator, FieldValidationInfo

# Inicializar colorama
init(autoreset=True)

load_dotenv()

# Cargar variables de entorno
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Construir la URL de la base de datos
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Configuración de la base de datos
try:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    print(Fore.GREEN + "Conexión exitosa a la base de datos PostgreSQL")
except Exception as e:
    print(Fore.RED + f"Error al conectar a la base de datos: {e}")

# Configuración de JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Modelos SQLAlchemy
class Usuario(Base):
    __tablename__ = "usuarios"
    __table_args__ = {"schema": "api"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nombre = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    es_admin = Column(Boolean, default=False)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    soft_delete = Column(Boolean, default=False)

    dispositivos = relationship("Dispositivo", back_populates="usuario")

# cambio 8
class Dispositivo(Base):
    __tablename__ = "dispositivos"
    __table_args__ = {"schema": "api"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey("api.usuarios.id", ondelete="CASCADE"))
    session_id = Column(String(255), unique=True)
    device_id = Column(String(255))
    device_name = Column(String(255), nullable=False, default="Dispositivo sin nombre") #cambio 20 y pico
    esta_online = Column(Boolean, default=False)
    ultimo_acceso = Column(DateTime(timezone=True), server_default=func.now())
    modelo = Column(String(100))
    sistema_operativo = Column(String(100))
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    soft_delete = Column(Boolean, default=False)

    usuario = relationship("Usuario", back_populates="dispositivos")
    notificaciones_dispositivo = relationship("NotificacionDispositivo", back_populates="dispositivo")

class Notificacion(Base):
    __tablename__ = "notificaciones"
    __table_args__ = {"schema": "api"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    titulo = Column(String(255), nullable=False)
    mensaje = Column(Text, nullable=False)
    imagen_url = Column(Text)
    dispositivos_objetivo = Column(ARRAY(UUID), default=None)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    soft_delete = Column(Boolean, default=False)

    notificaciones_dispositivo = relationship("NotificacionDispositivo", back_populates="notificacion")

class NotificacionDispositivo(Base):
    __tablename__ = "notificaciones_dispositivo"
    __table_args__ = {"schema": "api"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    notificacion_id = Column(UUID(as_uuid=True), ForeignKey("api.notificaciones.id", ondelete="CASCADE"))
    dispositivo_id = Column(UUID(as_uuid=True), ForeignKey("api.dispositivos.id", ondelete="CASCADE"))
    enviada = Column(Boolean, default=False)
    leida = Column(Boolean, default=False)
    sonando = Column(Boolean, default=False)
    fecha_envio = Column(DateTime(timezone=True))
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    soft_delete = Column(Boolean, default=False)

    notificacion = relationship("Notificacion", back_populates="notificaciones_dispositivo")
    dispositivo = relationship("Dispositivo", back_populates="notificaciones_dispositivo")

class ConfiguracionSonido(Base):
    __tablename__ = "configuracion_sonidos"
    __table_args__ = {"schema": "api"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey("api.usuarios.id", ondelete="CASCADE"))
    sonido = Column(String(100), default='default.mp3')
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    soft_delete = Column(Boolean, default=False)

    usuario = relationship("Usuario")

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    __table_args__ = {"schema": "api"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey("api.usuarios.id", ondelete="CASCADE"))
    token = Column(String(255), unique=True, nullable=False)
    fecha_expiracion = Column(DateTime(timezone=True), nullable=False)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    soft_delete = Column(Boolean, default=False)

    usuario = relationship("Usuario")

# Modelos Pydantic mejorados
class UsuarioCrear(BaseModel):
    nombre: str
    email: str
    password: str

class UsuarioSalida(BaseModel):
    id: uuid.UUID
    nombre: str
    email: str
    es_admin: bool

    class Config:
        from_attributes = True

class NuevaContrasena(BaseModel):
    nueva_contrasena: str

# cambio 7
class DispositivoCrear(BaseModel):
    device_id: str
    device_name: str
    modelo: Optional[str]
    sistema_operativo: Optional[str]

# cambio 7.1 cambio update 25/10/2024
class DispositivoSalida(BaseModel):
    id: uuid.UUID
    device_name: Optional[str]  # Permitir valores nulos
    esta_online: bool
    ultimo_acceso: datetime
    modelo: Optional[str]
    sistema_operativo: Optional[str]

    class Config:
        from_attributes = True

    @field_validator('device_name')
    def set_default_name(cls, v, info: FieldValidationInfo):
        return v or "Dispositivo sin nombre"

class NotificacionCrear(BaseModel):
    titulo: str
    mensaje: str
    imagen_url: Optional[str] = None
    dispositivos_objetivo: Optional[List[uuid.UUID]] = None  # Nuevo campo

class NotificacionSalida(BaseModel):
    id: uuid.UUID
    titulo: str
    mensaje: str
    imagen_url: str | None
    fecha_creacion: datetime

    class Config:
        from_attributes = True

# cambio 14 25/10/2024
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    session_id: str
    expires_in: int
    user_info: dict
    device_info: dict

    class Config:
        from_attributes = True

class TokenData(BaseModel):
    email: Optional[str] = None

class EstadoDispositivo(BaseModel):
    esta_online: bool

# Decoradores
def require_admin(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        current_user = None
        # Buscar current_user en los argumentos
        for arg in args:
            if isinstance(arg, Usuario):
                current_user = arg
                break
        if not current_user:
            for value in kwargs.values():
                if isinstance(value, Usuario):
                    current_user = value
                    break

        if not current_user or not current_user.es_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Se requieren privilegios de administrador para esta operación"
            )
        return await func(*args, **kwargs)
    return wrapper

# 

"""
    cambio 17 25/10/2024 
    Unificación de gestión de sesiones DB 
    Reemplaza tanto get_db() como get_db_session()
"""
@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Función helper para obtener DB en FastAPI endpoints
def get_db():
    with get_db_session() as db:
        yield db

# Configuración de OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Función de ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.start()
    scheduler.add_job(verificar_dispositivos_inactivos, IntervalTrigger(minutes=5))
    
    # Iniciar tarea de mantener vivas las conexiones WebSocket
    asyncio.create_task(manager.keep_alive())
    
    yield
    
    scheduler.shutdown()

# Crear una única instancia de FastAPI
app = FastAPI(lifespan=lifespan)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# cambio 9
# clase ConnectionManager para manejar reconexiones y asociar conexiones con usuarios
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # session_id -> WebSocket
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id

    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        await websocket.accept()
        
        # Si el usuario ya tiene una conexión activa, cerrarla
        if user_id in self.user_sessions:
            old_session = self.user_sessions[user_id]
            if old_session in self.active_connections:
                try:
                    old_websocket = self.active_connections[old_session]
                    await old_websocket.send_text(json.dumps({
                        "tipo": "sesion_cerrada",
                        "mensaje": "Se ha iniciado sesión en otro dispositivo"
                    }))
                    await old_websocket.close()
                except Exception as e:
                    logger.error(f"Error al cerrar conexión anterior: {str(e)}")
                del self.active_connections[old_session]
        
        self.active_connections[session_id] = websocket
        self.user_sessions[user_id] = session_id
        logger.info(f"Nueva conexión WebSocket establecida - Session ID: {session_id}, User ID: {user_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            # Encontrar y eliminar el user_id correspondiente
            user_id = next((uid for uid, sid in self.user_sessions.items() if sid == session_id), None)
            if user_id:
                del self.user_sessions[user_id]
            del self.active_connections[session_id]
            logger.info(f"Conexión WebSocket cerrada - Session ID: {session_id}")

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)
                logger.info(f"Mensaje enviado a session_id: {session_id}")
            else:
                logger.warning(f"Intento de enviar mensaje a sesión desconectada: {session_id}")
                self.disconnect(session_id)
        else:
            logger.warning(f"Sesión no encontrada: {session_id}")

    async def keep_alive(self):
        while True:
            for session_id, websocket in list(self.active_connections.items()):
                try:
                    await websocket.send_text(json.dumps({"tipo": "ping"}))
                except Exception as e:
                    logger.error(f"Error en keep-alive para session_id {session_id}: {str(e)}")
                    self.disconnect(session_id)
            await asyncio.sleep(30)

manager = ConnectionManager()

# Funciones de autenticación
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# cambio1
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    if "session_id" not in to_encode:
        to_encode["session_id"] = str(uuid.uuid4())  # Generar session_id si no existe
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, to_encode["session_id"]

def create_refresh_token(data: dict, db: Session):
    user_id = data["sub"]
    expires = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    token = os.urandom(64).hex()
    db_token = RefreshToken(usuario_id=user_id, token=token, fecha_expiracion=expires)
    db.add(db_token)
    db.commit()
    return token

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(Usuario).filter(Usuario.email == token_data.email, Usuario.soft_delete == False).first()
    if user is None:
        raise credentials_exception
    return user

# cambio 12
# Funcion que cierra las sesiones anteriores de un usuario y notifica a los dispositivos afectados.
async def cerrar_sesiones_anteriores(user_id: UUID, db: Session) -> Optional[Dispositivo]:
    try:
        dispositivo_anterior = db.query(Dispositivo).filter(
            Dispositivo.usuario_id == user_id,
            Dispositivo.esta_online == True,
            Dispositivo.soft_delete == False
        ).first()

        if dispositivo_anterior:
            current_timestamp = datetime.now(timezone.utc)
            
            mensaje = json.dumps({
                "tipo": "sesion_cerrada",
                "mensaje": "Se ha iniciado sesión en otro dispositivo",
                "timestamp": current_timestamp.isoformat()
            })

            # Intentar notificar con reintentos
            max_retries = 3
            retry_count = 0
            notification_sent = False

            while retry_count < max_retries and not notification_sent:
                try:
                    if dispositivo_anterior.session_id in manager.active_connections:
                        await manager.send_personal_message(
                            mensaje,
                            dispositivo_anterior.session_id
                        )
                        notification_sent = True
                        logger.info(
                            f"Notificación de cierre enviada al dispositivo anterior "
                            f"(session_id: {dispositivo_anterior.session_id})"
                        )
                    else:
                        break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.warning(
                            f"No se pudo notificar al dispositivo anterior después de "
                            f"{max_retries} intentos: {str(e)}"
                        )
                    await asyncio.sleep(0.5)

            # Actualizar estado del dispositivo
            dispositivo_anterior.esta_online = False
            dispositivo_anterior.session_id = None
            dispositivo_anterior.fecha_actualizacion = current_timestamp
            dispositivo_anterior.ultimo_acceso = current_timestamp
            
            return dispositivo_anterior

        return None
    except Exception as e:
        logger.error(f"Error al cerrar sesiones anteriores: {str(e)}")
        raise

# Manejadores de errores 
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_exception"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Excepción no manejada: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Se ha producido un error interno en el servidor.", "type": "server_error"}
    )


# Rutas de la API

# cambio 2 cambio 2.changes 25/10/2024
@app.post("/token", response_model=Token)
async def login_for_access_token(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        # Obtener datos del formulario
        form_data = await request.form()
        
        # Validar campos requeridos del formulario
        required_fields = ['username', 'password', 'device_id', 'device_name']
        missing_fields = [field for field in required_fields if not form_data.get(field)]
        
        if missing_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Campos requeridos faltantes",
                    "missing_fields": missing_fields
                }
            )

        # Validar y crear objeto DispositivoCrear
        try:
            device_info = DispositivoCrear(
                device_id=form_data.get('device_id'),
                device_name=form_data.get('device_name'),
                modelo=form_data.get('modelo'),
                sistema_operativo=form_data.get('sistema_operativo')
            )
        except ValidationError as e:
            logger.error(f"Error de validación en device_info: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Error de validación en la información del dispositivo",
                    "details": e.errors()
                }
            )

        # Validar credenciales
        user = db.query(Usuario).filter(
            Usuario.email == form_data.get('username'),
            Usuario.soft_delete == False
        ).first()
        
        if not user or not verify_password(form_data.get('password'), user.password):
            logger.warning(f"Intento de login fallido para el email: {form_data.get('username')}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales incorrectas"
            )

        # Generar nuevo session_id y obtener timestamp actual
        new_session_id = str(uuid.uuid4())
        current_timestamp = datetime.now(timezone.utc)

        # Cerrar sesión anterior si existe
        await cerrar_sesiones_anteriores(user.id, db)

        try:
            # Buscar o crear dispositivo
            dispositivo = db.query(Dispositivo).filter(
                Dispositivo.usuario_id == user.id,
                Dispositivo.device_id == device_info.device_id,
                Dispositivo.soft_delete == False
            ).first()

            if dispositivo:
                # Actualizar dispositivo existente
                dispositivo.session_id = new_session_id
                dispositivo.esta_online = True
                dispositivo.ultimo_acceso = current_timestamp
                dispositivo.fecha_actualizacion = current_timestamp
                if device_info.device_name:
                    dispositivo.device_name = device_info.device_name
                if device_info.modelo:
                    dispositivo.modelo = device_info.modelo
                if device_info.sistema_operativo:
                    dispositivo.sistema_operativo = device_info.sistema_operativo
                
                logger.info(f"Dispositivo actualizado: {device_info.device_id}")
            else:
                # Crear nuevo dispositivo
                dispositivo = Dispositivo(
                    usuario_id=user.id,
                    device_id=device_info.device_id,
                    device_name=device_info.device_name,
                    session_id=new_session_id,
                    modelo=device_info.modelo,
                    sistema_operativo=device_info.sistema_operativo,
                    esta_online=True,
                    ultimo_acceso=current_timestamp,
                    fecha_creacion=current_timestamp,
                    fecha_actualizacion=current_timestamp
                )
                db.add(dispositivo)
                logger.info(f"Nuevo dispositivo registrado: {device_info.device_id}")

            # Generar tokens con información adicional de seguridad
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token, _ = create_access_token(
                data={
                    "sub": user.email,
                    "session_id": new_session_id,
                    "device_id": device_info.device_id,
                    "iat": current_timestamp.timestamp(),
                    "login_timestamp": current_timestamp.isoformat()
                },
                expires_delta=access_token_expires
            )
            
            refresh_token = create_refresh_token(
                data={
                    "sub": str(user.id),
                    "device_id": device_info.device_id,
                    "session_id": new_session_id
                },
                db=db
            )

            # Commit de la transacción
            db.commit()

            # Registrar el evento de login exitoso
            logger.info(
                f"Login exitoso para {user.email} "
                f"(session_id: {new_session_id}, "
                f"device_id: {device_info.device_id})"
            )
            
            # Preparar respuesta
            response_data = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "session_id": new_session_id,
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "user_info": {
                    "id": str(user.id),
                    "email": user.email,
                    "nombre": user.nombre,
                    "es_admin": user.es_admin
                },
                "device_info": {
                    "device_id": device_info.device_id,
                    "device_name": device_info.device_name,
                    "session_created_at": current_timestamp.isoformat()
                }
            }

            return response_data

        except IntegrityError as e:
            db.rollback()
            logger.error(f"Error de integridad en la base de datos: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Error de integridad en la base de datos"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error inesperado en login: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno en el proceso de login"
        )
        
# cambio 2.1
# nuevo endpoint 
@app.get("/sesion-activa")
async def obtener_sesion_activa(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        dispositivo = db.query(Dispositivo).filter(
            Dispositivo.usuario_id == current_user.id,
            Dispositivo.esta_online == True,
            Dispositivo.soft_delete == False
        ).first()

        if not dispositivo:
            return {
                "activa": False,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        return {
            "activa": True,
            "session_id": dispositivo.session_id,
            "device_name": dispositivo.device_name,
            "device_id": dispositivo.device_id,
            "ultimo_acceso": dispositivo.ultimo_acceso.isoformat(),
            "modelo": dispositivo.modelo,
            "sistema_operativo": dispositivo.sistema_operativo,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error al obtener información de sesión: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error al obtener información de la sesión"
        )

# cambio 3... Update cambio 15 25/10/2024
@app.post("/token/refresh", response_model=Token)
async def refresh_token(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        form_data = await request.form()
        refresh_token = form_data.get('refresh_token')
        
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="refresh_token es requerido"
            )

        current_timestamp = datetime.now(timezone.utc)

        try:
            # Verificar el token de refresco
            db_token = db.query(RefreshToken).filter(
                RefreshToken.token == refresh_token,
                RefreshToken.soft_delete == False
            ).first()
            
            if not db_token:
                logger.warning(f"Intento de refresh con token inválido: {refresh_token[:10]}...")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token de refresco inválido"
                )

            # Convertir fecha_expiracion a UTC si no lo está
            if db_token.fecha_expiracion.tzinfo is None:
                expiracion = db_token.fecha_expiracion.replace(tzinfo=timezone.utc)
            else:
                expiracion = db_token.fecha_expiracion

            # Verificar expiración
            if expiracion < current_timestamp:
                logger.warning(f"Intento de refresh con token expirado para usuario_id: {db_token.usuario_id}")
                db_token.soft_delete = True
                db.commit()
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token de refresco expirado"
                )

            # Obtener usuario
            user = db.query(Usuario).filter(
                Usuario.id == db_token.usuario_id,
                Usuario.soft_delete == False
            ).first()
            
            if not user:
                logger.warning(f"Usuario no encontrado para refresh token: {db_token.usuario_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Usuario no encontrado"
                )

            # Obtener el dispositivo activo actual
            dispositivo = db.query(Dispositivo).filter(
                Dispositivo.usuario_id == user.id,
                Dispositivo.esta_online == True,
                Dispositivo.soft_delete == False
            ).first()

            if not dispositivo or not dispositivo.session_id:
                logger.warning(f"No hay sesión activa para el usuario: {user.email}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No hay sesión activa"
                )

            # Actualizar último acceso del dispositivo
            dispositivo.ultimo_acceso = current_timestamp
            dispositivo.fecha_actualizacion = current_timestamp

            # Generar nuevo access token manteniendo el mismo session_id
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token, _ = create_access_token(
                data={
                    "sub": user.email,
                    "session_id": dispositivo.session_id,
                    "device_id": dispositivo.device_id,
                    "iat": current_timestamp.timestamp(),
                    "refresh_timestamp": current_timestamp.isoformat()
                },
                expires_delta=access_token_expires
            )

            # Generar nuevo refresh token
            new_refresh_token = create_refresh_token(
                data={
                    "sub": str(user.id),
                    "device_id": dispositivo.device_id,
                    "session_id": dispositivo.session_id
                },
                db=db
            )
            
            # Invalidar el token anterior
            db_token.soft_delete = True
            db_token.fecha_actualizacion = current_timestamp

            # Limpiar tokens antiguos del usuario
            db.query(RefreshToken).filter(
                RefreshToken.usuario_id == user.id,
                RefreshToken.fecha_expiracion < current_timestamp,
                RefreshToken.soft_delete == False
            ).update({
                "soft_delete": True,
                "fecha_actualizacion": current_timestamp
            })

            db.commit()

            logger.info(
                f"Token refrescado exitosamente para {user.email} "
                f"(session_id: {dispositivo.session_id}, "
                f"device_id: {dispositivo.device_id})"
            )
            
            return {
                "access_token": access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "session_id": dispositivo.session_id,
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "user_info": {
                    "id": str(user.id),
                    "email": user.email,
                    "nombre": user.nombre,
                    "es_admin": user.es_admin
                },
                "device_info": {
                    "device_id": dispositivo.device_id,
                    "device_name": dispositivo.device_name,
                    "session_created_at": dispositivo.ultimo_acceso.isoformat()
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Error en la actualización de tokens: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error al procesar la renovación del token"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error en refresh token: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al renovar el token"
        )

# cambio 14
# endpoint deleted 25/10/2024
# Manejo de tokens de dispositivo
# @app.put("/dispositivos/{dispositivo_id}/token")
# Ya no es relevante con la nueva arquitectura de sesiones que estamos usando

@app.post("/usuarios", response_model=UsuarioSalida)
async def crear_usuario(usuario: UsuarioCrear, db: Session = Depends(get_db)):
    try:
        db_usuario = Usuario(
            nombre=usuario.nombre,
            email=usuario.email,
            password=get_password_hash(usuario.password)
        )
        db.add(db_usuario)
        db.commit()
        db.refresh(db_usuario)
        logger.info(f"Usuario creado: {usuario.email}")
        return db_usuario
    except IntegrityError as e:
        db.rollback()
        if "usuarios_email_key" in str(e.orig):
            logger.info(f"Intento de crear usuario con email duplicado: {usuario.email}")
            raise HTTPException(status_code=400, detail="El email ya está registrado")
        else:
            logger.error(f"Error de integridad al crear usuario: {str(e)}")
            raise HTTPException(status_code=400, detail="Error al crear usuario")
    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear usuario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al crear usuario")

@app.get("/usuarios/me", response_model=UsuarioSalida)
async def leer_usuario_actual(current_user: Usuario = Depends(get_current_user)):
    logger.info(f"Usuario {current_user.email} accedió a su perfil")
    return current_user

# nuevo endpoint 25/10/2024
@app.get("/usuarios", response_model=List[UsuarioSalida])
@require_admin  # Solo los administradores pueden ver la lista de usuarios
async def obtener_usuarios(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        usuarios = db.query(Usuario).filter(
            Usuario.soft_delete == False
        ).all()
        
        logger.info(f"Lista de usuarios solicitada por el administrador {current_user.email}")
        return usuarios
        
    except Exception as e:
        logger.error(f"Error al obtener lista de usuarios: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener la lista de usuarios"
        )

# cambio 16 25/10/2024
# este endpoint ya no es necesario pues la entrada de dispositivos la hacemos en el login
# @app.post("/dispositivos", response_model=DispositivoSalida)

# cambio 11
# Ruta para que los dispositivos informen su estado
@app.post("/dispositivos/{session_id}/ping")
async def dispositivo_ping(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        dispositivo = db.query(Dispositivo).filter(
            Dispositivo.session_id == session_id,
            Dispositivo.usuario_id == current_user.id,
            Dispositivo.soft_delete == False
        ).first()

        if not dispositivo:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        dispositivo.esta_online = True
        dispositivo.ultimo_acceso = datetime.now(timezone.utc)
        db.commit()
        
        background_tasks.add_task(enviar_notificaciones_pendientes, session_id)
        
        return {
            "mensaje": "Estado del dispositivo actualizado",
            "ultimo_acceso": dispositivo.ultimo_acceso.isoformat()
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error en ping: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al actualizar el estado del dispositivo")

# Ruta nueva para reintento manual de envío de notificaciones
@app.post("/notificaciones/{notificacion_id}/reenviar")
async def reenviar_notificacion(
    notificacion_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.es_admin:
        raise HTTPException(status_code=403, detail="No tienes permiso para reenviar notificaciones")
    
    with get_db_session() as db:
        try:
            notificacion = db.query(Notificacion).filter(
                Notificacion.id == notificacion_id,
                Notificacion.soft_delete == False
            ).first()
            
            if not notificacion:
                raise HTTPException(status_code=404, detail="Notificación no encontrada")
            
            # Reiniciar el estado de envío para todas las asociaciones de esta notificación
            db.query(NotificacionDispositivo).filter(
                NotificacionDispositivo.notificacion_id == notificacion_id,
                NotificacionDispositivo.soft_delete == False
            ).update({"enviada": False, "fecha_envio": None})
            
            db.commit()
            
            background_tasks.add_task(enviar_notificaciones, str(notificacion_id))
            
            return {"mensaje": "Reenvío de notificación iniciado"}
        except Exception as e:
            logger.error(f"Error al reenviar notificación: {str(e)}")
            raise HTTPException(status_code=500, detail="Error al reenviar la notificación")

# cambio 5... cambio 18 aplicacion del decorador 25/10/2024
@app.post("/notificaciones", response_model=NotificacionSalida)
@require_admin
async def crear_notificacion(
    notificacion: NotificacionCrear,
    background_tasks: BackgroundTasks,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        db_notificacion = Notificacion(**notificacion.model_dump())
        db.add(db_notificacion)
        db.flush()

        # Obtener dispositivos activos
        dispositivos = db.query(Dispositivo).filter(
            Dispositivo.esta_online == True,
            Dispositivo.soft_delete == False,
            Dispositivo.session_id.isnot(None)
        ).all()

        for dispositivo in dispositivos:
            notif_dispositivo = NotificacionDispositivo(
                notificacion_id=db_notificacion.id,
                dispositivo_id=dispositivo.id,
                leida=False,
                sonando=True
            )
            db.add(notif_dispositivo)

        db.commit()
        db.refresh(db_notificacion)
        
        logger.info(f"Notificación {db_notificacion.id} creada y será enviada a {len(dispositivos)} dispositivos activos")
        
        # Iniciar el envío de notificaciones en segundo plano
        background_tasks.add_task(enviar_notificaciones, str(db_notificacion.id))
        
        return db_notificacion

    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear notificación: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al crear la notificación")

@app.get("/notificaciones", response_model=List[NotificacionSalida])
async def leer_notificaciones(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    try:
        dispositivos = db.query(Dispositivo).filter(Dispositivo.usuario_id == current_user.id, Dispositivo.soft_delete == False).all()
        dispositivo_ids = [d.id for d in dispositivos]
        notificaciones = db.query(Notificacion).join(NotificacionDispositivo).filter(
            NotificacionDispositivo.dispositivo_id.in_(dispositivo_ids),
            Notificacion.soft_delete == False
        ).order_by(Notificacion.fecha_creacion.desc()).offset(skip).limit(limit).all()
        logger.info(f"Usuario {current_user.email} accedió a las notificaciones (paginadas)")
        return notificaciones
    except Exception as e:
        logger.error(f"Error al leer notificaciones paginadas: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener las notificaciones")

# cambio 19 aplicacion del decorador require_admin 25/10/2024
@app.get("/notificaciones/estadisticas")
@require_admin
async def obtener_estadisticas_notificaciones(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        total_notificaciones = db.query(func.count(Notificacion.id)).filter(
            Notificacion.soft_delete == False
        ).scalar()
        
        notificaciones_enviadas = db.query(func.count(NotificacionDispositivo.id)).filter(
            NotificacionDispositivo.enviada == True,
            NotificacionDispositivo.soft_delete == False
        ).scalar()
        
        notificaciones_leidas = db.query(func.count(NotificacionDispositivo.id)).filter(
            NotificacionDispositivo.leida == True,
            NotificacionDispositivo.soft_delete == False
        ).scalar()
        
        dispositivos_activos = db.query(func.count(Dispositivo.id)).filter(
            Dispositivo.esta_online == True,
            Dispositivo.soft_delete == False
        ).scalar()
        
        return {
            "total_notificaciones": total_notificaciones,
            "notificaciones_enviadas": notificaciones_enviadas,
            "notificaciones_leidas": notificaciones_leidas,
            "dispositivos_activos": dispositivos_activos
        }
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de notificaciones: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener estadísticas")

# cambio 6
# nuevo endpoint
# cambio 20 aplicacion del decorador require_admin 25/10/2024
@app.get("/notificaciones/{notificacion_id}/estado")
@require_admin
async def obtener_estado_notificacion(
    notificacion_id: uuid.UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Obtener estadísticas de entrega
        estados = db.query(
            func.count(NotificacionDispositivo.id).label('total'),
            func.sum(case((NotificacionDispositivo.enviada == True, 1), else_=0)).label('enviadas'),
            func.sum(case((NotificacionDispositivo.leida == True, 1), else_=0)).label('leidas')
        ).filter(
            NotificacionDispositivo.notificacion_id == notificacion_id,
            NotificacionDispositivo.soft_delete == False
        ).first()

        return {
            "total_dispositivos": estados.total or 0,
            "enviadas": estados.enviadas or 0,
            "leidas": estados.leidas or 0
        }

    except Exception as e:
        logger.error(f"Error al obtener estado de notificación: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener el estado de la notificación")

@app.put("/notificaciones/{notificacion_id}/leer")
async def marcar_notificacion_como_leida(
    notificacion_id: uuid.UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    with get_db_session() as db:
        try:
            notificacion = db.query(Notificacion).filter(
                Notificacion.id == notificacion_id,
                Notificacion.soft_delete == False
            ).first()
            
            if not notificacion:
                raise HTTPException(status_code=404, detail="Notificación no encontrada")

            dispositivos = db.query(Dispositivo).filter(Dispositivo.usuario_id == current_user.id, Dispositivo.soft_delete == False).all()
            dispositivo_ids = [d.id for d in dispositivos]
            
            result = db.query(NotificacionDispositivo).filter(
                NotificacionDispositivo.notificacion_id == notificacion_id,
                NotificacionDispositivo.dispositivo_id.in_(dispositivo_ids),
                NotificacionDispositivo.soft_delete == False
            ).update({"leida": True, "sonando": False})
            
            if result == 0:
                raise HTTPException(status_code=404, detail="Notificación no asociada a ninguno de tus dispositivos")
            
            logger.info(f"Notificación {notificacion_id} marcada como leída para todos los dispositivos del usuario {current_user.email}")
            return {"mensaje": "Notificación marcada como leída y detenida para todos tus dispositivos"}
        except Exception as e:
            logger.error(f"Error al marcar notificación como leída: {str(e)}")
            raise HTTPException(status_code=500, detail="Error al marcar la notificación como leída")

# Endpoint para notificaciones no leídas
@app.get("/notificaciones/no-leidas", response_model=List[NotificacionSalida])
async def obtener_notificaciones_no_leidas(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    try:
        # Obtener los IDs de los dispositivos del usuario
        dispositivos = db.query(Dispositivo).filter(Dispositivo.usuario_id == current_user.id, Dispositivo.soft_delete == False).all()
        dispositivo_ids = [d.id for d in dispositivos]

        # Consulta para obtener notificaciones no leídas
        notificaciones = db.query(Notificacion).join(NotificacionDispositivo).filter(
            NotificacionDispositivo.dispositivo_id.in_(dispositivo_ids),
            NotificacionDispositivo.leida == False,
            Notificacion.soft_delete == False,
            NotificacionDispositivo.soft_delete == False
        ).order_by(Notificacion.fecha_creacion.desc()).offset(skip).limit(limit).all()

# Log para depuración
        logger.info(f"Usuario {current_user.email} solicitó notificaciones no leídas. Encontradas: {len(notificaciones)}")
        
        return notificaciones
    except Exception as e:
        logger.error(f"Error al obtener notificaciones no leídas: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener las notificaciones no leídas")

@app.post("/usuarios/configuracion-sonido")
async def establecer_configuracion_sonido(
    sonido: str = Body(..., embed=True),
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        configuracion = db.query(ConfiguracionSonido).filter(ConfiguracionSonido.usuario_id == current_user.id).first()
        if configuracion:
            configuracion.sonido = sonido
        else:
            nueva_configuracion = ConfiguracionSonido(usuario_id=current_user.id, sonido=sonido)
            db.add(nueva_configuracion)
        db.commit()
        logger.info(f"Configuración de sonido actualizada para el usuario {current_user.email}")
        return {"mensaje": "Configuración de sonido actualizada correctamente"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar configuración de sonido: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al actualizar la configuración de sonido")

@app.get("/usuarios/configuracion-sonido")
async def obtener_configuracion_sonido(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        configuracion = db.query(ConfiguracionSonido).filter(ConfiguracionSonido.usuario_id == current_user.id).first()
        if not configuracion:
            return {"sonido": "default.mp3"}
        return {"sonido": configuracion.sonido}
    except Exception as e:
        logger.error(f"Error al obtener configuración de sonido: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener la configuración de sonido")

# cambio 21 aplicacion del decorador require_admin 25/10/2024
@app.get("/dispositivos", response_model=List[DispositivoSalida])
@require_admin
async def leer_dispositivos(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        dispositivos = db.query(Dispositivo).filter(Dispositivo.soft_delete == False).all()
        logger.info(f"Usuario administrador {current_user.email} accedió a los dispositivos")
        return dispositivos
    except Exception as e:
        logger.error(f"Error al leer dispositivos: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener los dispositivos")

@app.put("/dispositivos/{dispositivo_id}/online")
async def actualizar_estado_dispositivo(
    dispositivo_id: uuid.UUID,
    estado: EstadoDispositivo,
    background_tasks: BackgroundTasks,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        dispositivo = db.query(Dispositivo).filter(Dispositivo.id == dispositivo_id, Dispositivo.soft_delete == False).first()
        if not dispositivo or dispositivo.usuario_id != current_user.id:
            raise HTTPException(status_code=404, detail="Dispositivo no encontrado")

        dispositivo.esta_online = estado.esta_online
        dispositivo.ultimo_acceso = datetime.now(timezone.utc)
        db.commit()

        if estado.esta_online:
            background_tasks.add_task(enviar_notificaciones_pendientes, str(dispositivo_id))

        logger.info(f"Estado del dispositivo {dispositivo_id} actualizado a {'en línea' if estado.esta_online else 'fuera de línea'} por el usuario {current_user.email}")
        return {"mensaje": f"Estado del dispositivo actualizado a {'en línea' if estado.esta_online else 'fuera de línea'}"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar estado del dispositivo: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al actualizar el estado del dispositivo")

# cambio 22 aplicacion del decorador require_admin 25/10/2024
@app.delete("/usuarios/{usuario_id}")
@require_admin
async def eliminar_usuario(
    usuario_id: uuid.UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id, Usuario.soft_delete == False).first()
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        usuario.soft_delete = True
        db.commit()
        logger.info(f"Usuario {usuario_id} eliminado por el administrador {current_user.email}")
        return {"mensaje": "Usuario eliminado correctamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar usuario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al eliminar el usuario")

@app.put("/usuarios/cambiar_contrasena", response_model=UsuarioSalida)
async def cambiar_contrasena(
    nueva_contrasena: NuevaContrasena,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        current_user.password = get_password_hash(nueva_contrasena.nueva_contrasena)
        db.commit()
        db.refresh(current_user)
        logger.info(f"Usuario {current_user.email} cambió su contraseña")
        return current_user
    except Exception as e:
        db.rollback()
        logger.error(f"Error al cambiar contraseña: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al cambiar la contraseña")

# cambio 23 aplicacion del decorador require_admin 25/10/2024
@app.put("/usuarios/{usuario_id}/restablecer_contrasena", response_model=UsuarioSalida)
@require_admin
async def restablecer_contrasena(
    usuario_id: uuid.UUID,
    nueva_contrasena: NuevaContrasena,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id, Usuario.soft_delete == False).first()
        if not usuario:
            logger.warning(f"Intento de restablecer contraseña para usuario no existente: {usuario_id}")
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        logger.info(f"Iniciando restablecimiento de contraseña para el usuario {usuario.email}")
        usuario.password = get_password_hash(nueva_contrasena.nueva_contrasena)
        db.commit()
        db.refresh(usuario)
        logger.info(f"Administrador {current_user.email} restableció la contraseña para el usuario {usuario.email}")
        return usuario
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al restablecer contraseña para {usuario.email if usuario else 'usuario desconocido'}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al restablecer la contraseña")
        
@app.delete("/dispositivos/{dispositivo_id}")
async def eliminar_dispositivo(
    dispositivo_id: uuid.UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        dispositivo = db.query(Dispositivo).filter(Dispositivo.id == dispositivo_id, Dispositivo.soft_delete == False).first()
        if not dispositivo or (dispositivo.usuario_id != current_user.id and not current_user.es_admin):
            raise HTTPException(status_code=404, detail="Dispositivo no encontrado")
        dispositivo.soft_delete = True
        db.commit()
        logger.info(f"Dispositivo {dispositivo_id} eliminado por el usuario {current_user.email}")
        return {"mensaje": "Dispositivo eliminado correctamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar dispositivo: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al eliminar el dispositivo")

# cambio 24 aplicacion del decorador require_admin 25/10/2024
@app.delete("/notificaciones/{notificacion_id}")
@require_admin
async def eliminar_notificacion(
    notificacion_id: uuid.UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        notificacion = db.query(Notificacion).filter(Notificacion.id == notificacion_id, Notificacion.soft_delete == False).first()
        if not notificacion:
            raise HTTPException(status_code=404, detail="Notificación no encontrada")
        notificacion.soft_delete = True
        db.commit()
        logger.info(f"Notificación {notificacion_id} eliminada por el administrador {current_user.email}")
        return {"mensaje": "Notificación eliminada correctamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar notificación: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al eliminar la notificación")

# cambio 10
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, token: str = Query(...)):
    db = SessionLocal()
    try:
        # Validar el token
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            token_session_id = payload.get("session_id")
            
            if not email or not token_session_id or token_session_id != session_id:
                logger.warning(f"WebSocket conexión rechazada: token inválido o session_id no coincide")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
                
            user = db.query(Usuario).filter(
                Usuario.email == email,
                Usuario.soft_delete == False
            ).first()
            
            if not user:
                logger.warning(f"WebSocket conexión rechazada: usuario no encontrado")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            # Verificar el dispositivo asociado a la sesión
            dispositivo = db.query(Dispositivo).filter(
                Dispositivo.session_id == session_id,
                Dispositivo.usuario_id == user.id,
                Dispositivo.soft_delete == False
            ).first()
            
            if not dispositivo:
                logger.warning(f"WebSocket conexión rechazada: sesión no encontrada")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            # Actualizar estado del dispositivo
            dispositivo.esta_online = True
            dispositivo.ultimo_acceso = datetime.now(timezone.utc)
            db.commit()
            
            # Establecer conexión WebSocket
            await manager.connect(websocket, session_id, str(user.id))
            
            # Iniciar un bucle de ping-pong en segundo plano
            ping_task = asyncio.create_task(keep_connection_alive(websocket))
            
            try:
                while True:
                    # Usar un timeout para el receive_text
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                        data_json = json.loads(data)
                        
                        if data_json.get("tipo") == "ping":
                            await websocket.send_text(json.dumps({"tipo": "pong"}))
                            # Actualizar último acceso
                            dispositivo.ultimo_acceso = datetime.now(timezone.utc)
                            db.commit()
                        
                    except asyncio.TimeoutError:
                        # Verificar si la conexión sigue activa
                        if websocket.client_state == WebSocketState.DISCONNECTED:
                            raise WebSocketDisconnect()
                        continue
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket desconectado para session_id {session_id}")
            finally:
                # Cancelar la tarea de ping
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
                
                # Actualizar estado del dispositivo y desconectar
                dispositivo.esta_online = False
                db.commit()
                manager.disconnect(session_id)
                    
        except jwt.PyJWTError as e:
            logger.warning(f"WebSocket conexión rechazada: error en token - {str(e)}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            
    except Exception as e:
        logger.error(f"Error en WebSocket para session_id {session_id}: {str(e)}")
        if session_id in manager.active_connections:
            manager.disconnect(session_id)
        
    finally:
        db.close()

async def keep_connection_alive(websocket: WebSocket):
    """Mantiene la conexión WebSocket activa mediante ping-pong periódicos"""
    while True:
        try:
            if websocket.client_state == WebSocketState.DISCONNECTED:
                break
            await websocket.send_text(json.dumps({"tipo": "ping"}))
            await asyncio.sleep(30)  # Enviar ping cada 30 segundos
        except Exception as e:
            logger.error(f"Error en keep_connection_alive: {str(e)}")
            break
# Funciones auxiliares

# cambio 4
async def enviar_notificaciones(notificacion_id: str):
    with get_db_session() as db:
        try:
            notificacion = db.query(Notificacion).filter(
                Notificacion.id == uuid.UUID(notificacion_id),
                Notificacion.soft_delete == False
            ).first()

            if not notificacion:
                logger.warning(f"Notificación {notificacion_id} no encontrada")
                return

            # Obtener todos los usuarios objetivo y sus dispositivos activos
            usuarios_dispositivos = db.query(Usuario, Dispositivo).join(
                Dispositivo,
                Usuario.id == Dispositivo.usuario_id
            ).filter(
                Dispositivo.esta_online == True,
                Dispositivo.soft_delete == False,
                Dispositivo.session_id.isnot(None)  # Solo dispositivos con sesión activa
            ).all()

            notificaciones_enviadas = 0
            for usuario, dispositivo in usuarios_dispositivos:
                # Crear o actualizar la relación notificación-dispositivo
                notif_dispositivo = db.query(NotificacionDispositivo).filter(
                    NotificacionDispositivo.notificacion_id == notificacion.id,
                    NotificacionDispositivo.dispositivo_id == dispositivo.id
                ).first()

                if not notif_dispositivo:
                    notif_dispositivo = NotificacionDispositivo(
                        notificacion_id=notificacion.id,
                        dispositivo_id=dispositivo.id,
                        leida=False,
                        sonando=True
                    )
                    db.add(notif_dispositivo)

                # Obtener configuración de sonido del usuario
                configuracion_sonido = db.query(ConfiguracionSonido).filter(
                    ConfiguracionSonido.usuario_id == usuario.id
                ).first()
                sonido = configuracion_sonido.sonido if configuracion_sonido else "default.mp3"

                # Preparar el mensaje
                mensaje = json.dumps({
                    "tipo": "nueva_notificacion",
                    "notificacion": {
                        "id": str(notificacion.id),
                        "titulo": notificacion.titulo,
                        "mensaje": notificacion.mensaje,
                        "imagen_url": notificacion.imagen_url,
                        "sonido": sonido
                    }
                })

                try:
                    # Enviar la notificación usando el session_id
                    if dispositivo.session_id in manager.active_connections:
                        await manager.send_personal_message(mensaje, dispositivo.session_id)
                        notif_dispositivo.enviada = True
                        notif_dispositivo.fecha_envio = datetime.now(timezone.utc)
                        notificaciones_enviadas += 1
                        logger.info(f"Notificación enviada al dispositivo {dispositivo.device_name} "
                                  f"(session_id: {dispositivo.session_id}) del usuario {usuario.email}")
                except Exception as e:
                    logger.error(f"Error al enviar notificación al dispositivo {dispositivo.id}: {str(e)}")

            db.commit()
            logger.info(f"Resumen de envío para notificación {notificacion_id}:")
            logger.info(f"Total de notificaciones enviadas: {notificaciones_enviadas}")

        except Exception as e:
            logger.error(f"Error en el proceso de envío de notificaciones: {str(e)}")
            raise

async def enviar_notificaciones_pendientes(dispositivo_id: str):
    with get_db_session() as db:
        try:
            notificaciones_pendientes = db.query(NotificacionDispositivo).options(
                joinedload(NotificacionDispositivo.notificacion)
            ).filter(
                NotificacionDispositivo.dispositivo_id == uuid.UUID(dispositivo_id),
                NotificacionDispositivo.enviada == False,
                NotificacionDispositivo.soft_delete == False
            ).all()

            for notif in notificaciones_pendientes:
                usuario = db.query(Usuario).filter(Usuario.id == notif.dispositivo.usuario_id).first()
                configuracion_sonido = db.query(ConfiguracionSonido).filter(ConfiguracionSonido.usuario_id == usuario.id).first()
                sonido = configuracion_sonido.sonido if configuracion_sonido else "default.mp3"

                mensaje = json.dumps({
                    "tipo": "notificacion_pendiente",
                    "notificacion": {
                        "id": str(notif.notificacion.id),
                        "titulo": notif.notificacion.titulo,
                        "mensaje": notif.notificacion.mensaje,
                        "imagen_url": notif.notificacion.imagen_url,
                        "sonido": sonido
                    }
                })
                try:
                    await manager.send_personal_message(mensaje, dispositivo_id)
                    notif.enviada = True
                    notif.fecha_envio = datetime.now(timezone.utc)
                except Exception as e:
                    logger.error(f"Error al enviar notificación pendiente al dispositivo {dispositivo_id}: {str(e)}")

            db.commit()
            logger.info(f"Notificaciones pendientes enviadas para el dispositivo {dispositivo_id}")
        except Exception as e:
            logger.error(f"Error inesperado al enviar notificaciones pendientes: {str(e)}")
            raise

# cambio 13 25/10/2024
def verificar_dispositivos_inactivos():
    with get_db_session() as db:
        try:
            tiempo_limite = datetime.now(timezone.utc) - timedelta(minutes=15)
            dispositivos_inactivos = db.query(Dispositivo).filter(
                Dispositivo.esta_online == True,
                Dispositivo.ultimo_acceso < tiempo_limite,
                Dispositivo.soft_delete == False
            ).all()

            for dispositivo in dispositivos_inactivos:
                dispositivo.esta_online = False
                # Intentar cerrar la conexión WebSocket si existe
                session_id = manager.user_sessions.get(str(dispositivo.usuario_id))
                if session_id:
                    websocket = manager.active_connections.get(session_id)
                    if websocket and websocket.client_state == WebSocketState.CONNECTED:
                        asyncio.create_task(websocket.close(code=status.WS_1000_NORMAL_CLOSURE))

            db.commit()
            logger.info(f"Dispositivos inactivos actualizados: {len(dispositivos_inactivos)}")
        except Exception as e:
            logger.error(f"Error inesperado al verificar dispositivos inactivos: {str(e)}")
            raise

def verificar_y_corregir_asociaciones(db: Session):
    try:
        # Obtener todas las notificaciones y dispositivos activos
        notificaciones = db.query(Notificacion).filter(Notificacion.soft_delete == False).all()
        dispositivos = db.query(Dispositivo).filter(Dispositivo.soft_delete == False).all()

        for notificacion in notificaciones:
            for dispositivo in dispositivos:
                # Verificar si ya existe una asociación
                existe_asociacion = db.query(NotificacionDispositivo).filter(
                    NotificacionDispositivo.notificacion_id == notificacion.id,
                    NotificacionDispositivo.dispositivo_id == dispositivo.id,
                    NotificacionDispositivo.soft_delete == False
                ).first()

                # Si no existe, crear la asociación
                if not existe_asociacion:
                    nueva_asociacion = NotificacionDispositivo(
                        notificacion_id=notificacion.id,
                        dispositivo_id=dispositivo.id,
                        leida=False
                    )
                    db.add(nueva_asociacion)

        db.commit()
        logger.info("Verificación y corrección de asociaciones completada")
    except Exception as e:
        db.rollback()
        logger.error(f"Error al verificar y corregir asociaciones: {str(e)}")
        raise

# Script de migración para actualizar registros existentes
def actualizar_nombres_dispositivos():
    with get_db_session() as db:
        try:
            db.query(Dispositivo).filter(
                Dispositivo.device_name.is_(None),
                Dispositivo.soft_delete == False
            ).update(
                {"device_name": "Dispositivo sin nombre"}, 
                synchronize_session=False
            )
            db.commit()
            logger.info("Nombres de dispositivos actualizados correctamente")
        except Exception as e:
            db.rollback()
            logger.error(f"Error al actualizar nombres de dispositivos: {str(e)}")
            raise

if __name__ == "__main__":
    import uvicorn
    APP_HOST = os.getenv("APP_HOST")
    APP_PORT = int(os.getenv("APP_PORT"))
    
    # Verificar y corregir asociaciones al inicio
    with SessionLocal() as db:
        verificar_y_corregir_asociaciones(db)
        actualizar_nombres_dispositivos()  
    
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
