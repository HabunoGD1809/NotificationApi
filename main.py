# Librerías estándar de Python
import asyncio
import os
import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from contextlib import contextmanager, asynccontextmanager

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
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState

# SQLAlchemy (ORM para bases de datos)
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, Session, joinedload, relationship

# Pydantic (para validación de datos en FastAPI)
from pydantic import BaseModel, EmailStr

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

class Dispositivo(Base):
    __tablename__ = "dispositivos"
    __table_args__ = {"schema": "api"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    usuario_id = Column(UUID(as_uuid=True), ForeignKey("api.usuarios.id", ondelete="CASCADE"))
    token = Column(String(255), unique=True, nullable=False)
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

class DispositivoCrear(BaseModel):
    token: str
    modelo: str
    sistema_operativo: str

class DispositivoSalida(BaseModel):
    id: uuid.UUID
    token: str
    esta_online: bool
    ultimo_acceso: datetime
    modelo: str
    sistema_operativo: str

    class Config:
        from_attributes = True

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

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class EstadoDispositivo(BaseModel):
    esta_online: bool

# Función para obtener la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Context manager para transacciones
@contextmanager
def get_db_transaction():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

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

# clase ConnectionManager para manejar reconexiones y asociar conexiones con usuarios
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket, client_id: str, user_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.user_connections[user_id] = client_id
        logger.info(f"Cliente {client_id} conectado para el usuario {user_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            for user_id, conn_id in self.user_connections.items():
                if conn_id == client_id:
                    del self.user_connections[user_id]
                    break
            logger.info(f"Cliente {client_id} desconectado")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)
                logger.info(f"Mensaje enviado al cliente {client_id}")
            else:
                logger.warning(f"Intento de enviar mensaje a cliente desconectado {client_id}")
        else:
            logger.warning(f"Cliente {client_id} no encontrado para enviar mensaje")

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            if connection.client_state == WebSocketState.CONNECTED:
                await connection.send_text(message)
        logger.info("Mensaje de difusión enviado")

    async def keep_alive(self):
        while True:
            for client_id, websocket in list(self.active_connections.items()):
                try:
                    await websocket.send_text("ping")
                except WebSocketDisconnect:
                    self.disconnect(client_id)
                except Exception as e:
                    logger.error(f"Error al enviar ping a {client_id}: {str(e)}")
                    self.disconnect(client_id)
            await asyncio.sleep(30)  # Enviar ping cada 30 segundos

manager = ConnectionManager()

# Funciones de autenticación
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        user = db.query(Usuario).filter(Usuario.email == form_data.username, Usuario.soft_delete == False).first()
        if not user:
            logger.warning(f"Intento de login fallido: usuario no encontrado - {form_data.username}")
            raise HTTPException(status_code=401, detail="Credenciales incorrectas")
        
        if not verify_password(form_data.password, user.password):
            logger.warning(f"Intento de login fallido: contraseña incorrecta - {form_data.username}")
            raise HTTPException(status_code=401, detail="Credenciales incorrectas")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
        refresh_token = create_refresh_token(data={"sub": str(user.id)}, db=db)
        
        logger.info(f"Usuario {user.email} ha iniciado sesión exitosamente")
        return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error inesperado en el proceso de login: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno en el proceso de login")

@app.post("/token/refresh", response_model=Token)
async def refresh_token(refresh_token: str = Body(..., embed=True), db: Session = Depends(get_db)):
    try:
        db_token = db.query(RefreshToken).filter(RefreshToken.token == refresh_token, RefreshToken.soft_delete == False).first()
        if not db_token:
            raise HTTPException(status_code=400, detail="Token de refresco inválido")

        # Convertir la fecha de expiración a UTC si no lo está ya
        expiration_date = db_token.fecha_expiracion.replace(tzinfo=timezone.utc) if db_token.fecha_expiracion.tzinfo is None else db_token.fecha_expiracion
        
        if expiration_date < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Token de refresco expirado")

        user = db.query(Usuario).filter(Usuario.id == db_token.usuario_id, Usuario.soft_delete == False).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
        new_refresh_token = create_refresh_token(data={"sub": str(user.id)}, db=db)
        
        # Invalidar el token de refresco anterior
        db_token.soft_delete = True
        db.commit()

        logger.info(f"Token de refresco renovado para el usuario {user.email}")
        return {"access_token": access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error en refresh token: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al renovar el token")

# Manejo de tokens de dispositivo
@app.put("/dispositivos/{dispositivo_id}/token")
async def actualizar_token_dispositivo(
    dispositivo_id: uuid.UUID,
    nuevo_token: str = Body(..., embed=True),
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        dispositivo = db.query(Dispositivo).filter(Dispositivo.id == dispositivo_id, Dispositivo.usuario_id == current_user.id, Dispositivo.soft_delete == False).first()
        if not dispositivo:
            raise HTTPException(status_code=404, detail="Dispositivo no encontrado")
        dispositivo.token = nuevo_token
        db.commit()
        logger.info(f"Token actualizado para el dispositivo {dispositivo_id}")
        return {"mensaje": "Token de dispositivo actualizado correctamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar token de dispositivo: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al actualizar el token del dispositivo")

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

@app.post("/dispositivos", response_model=DispositivoSalida)
async def registrar_dispositivo(
    dispositivo: DispositivoCrear,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        db_dispositivo = Dispositivo(
            usuario_id=current_user.id,
            token=dispositivo.token,
            modelo=dispositivo.modelo,
            sistema_operativo=dispositivo.sistema_operativo
        )
        db.add(db_dispositivo)
        db.flush()  # Para obtener el ID del dispositivo

        # Crear entradas en NotificacionDispositivo para notificaciones existentes
        notificaciones_existentes = db.query(Notificacion).filter(Notificacion.soft_delete == False).all()
        for notificacion in notificaciones_existentes:
            notif_dispositivo = NotificacionDispositivo(
                notificacion_id=notificacion.id,
                dispositivo_id=db_dispositivo.id,
                leida=False
            )
            db.add(notif_dispositivo)

        db.commit()
        db.refresh(db_dispositivo)
        logger.info(f"Dispositivo {dispositivo.token} registrado para el usuario {current_user.email}")
        return db_dispositivo
    except IntegrityError as e:
        db.rollback()
        if "dispositivos_token_key" in str(e.orig):
            logger.info(f"Intento de registrar dispositivo con token duplicado: {dispositivo.token}")
            raise HTTPException(status_code=400, detail="El token del dispositivo ya está registrado")
        else:
            logger.error(f"Error de integridad al registrar dispositivo: {str(e)}")
            raise HTTPException(status_code=400, detail="Error al registrar dispositivo")
    except Exception as e:
        db.rollback()
        logger.error(f"Error al registrar dispositivo: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al registrar dispositivo")

# Ruta para que los dispositivos informen su estado
@app.post("/dispositivos/{dispositivo_id}/ping")
async def dispositivo_ping(
    dispositivo_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    with get_db_transaction() as db:
        try:
            dispositivo = db.query(Dispositivo).filter(Dispositivo.id == dispositivo_id, Dispositivo.usuario_id == current_user.id).first()
            if not dispositivo:
                raise HTTPException(status_code=404, detail="Dispositivo no encontrado")
            
            dispositivo.esta_online = True
            dispositivo.ultimo_acceso = datetime.now(timezone.utc)
            
            db.commit()
            
            background_tasks.add_task(enviar_notificaciones_pendientes, str(dispositivo_id))
            
            return {"mensaje": "Estado del dispositivo actualizado"}
        except Exception as e:
            logger.error(f"Error al actualizar estado del dispositivo: {str(e)}")
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
    
    with get_db_transaction() as db:
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

@app.post("/notificaciones", response_model=NotificacionSalida)
async def crear_notificacion(
    notificacion: NotificacionCrear,
    background_tasks: BackgroundTasks,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.es_admin:
        logger.warning(f"Usuario no autorizado {current_user.email} intentó crear una notificación")
        raise HTTPException(status_code=403, detail="No tienes permiso para crear notificaciones")
    
    try:
        db_notificacion = Notificacion(**notificacion.dict(exclude={'dispositivos_objetivo'}))
        db.add(db_notificacion)
        db.flush()

        if notificacion.dispositivos_objetivo:
            dispositivos = db.query(Dispositivo).filter(
                Dispositivo.id.in_(notificacion.dispositivos_objetivo),
                Dispositivo.soft_delete == False
            ).all()
        else:
            dispositivos = db.query(Dispositivo).filter(Dispositivo.soft_delete == False).all()

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
        
        logger.info(f"Notificación {db_notificacion.id} creada y asociada a {len(dispositivos)} dispositivos")
        
        background_tasks.add_task(enviar_notificaciones, str(db_notificacion.id))
        
        # Crear un diccionario con los datos de la notificación
        notificacion_dict = {
            "id": db_notificacion.id,
            "titulo": db_notificacion.titulo,
            "mensaje": db_notificacion.mensaje,
            "imagen_url": db_notificacion.imagen_url,
            "fecha_creacion": db_notificacion.fecha_creacion
        }
        
        return NotificacionSalida(**notificacion_dict)
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

@app.get("/notificaciones/estadisticas")
async def obtener_estadisticas_notificaciones(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.es_admin:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver estadísticas")
    
    try:
        total_notificaciones = db.query(func.count(Notificacion.id)).filter(Notificacion.soft_delete == False).scalar()
        
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

@app.put("/notificaciones/{notificacion_id}/leer")
async def marcar_notificacion_como_leida(
    notificacion_id: uuid.UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    with get_db_transaction() as db:
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

@app.get("/dispositivos", response_model=List[DispositivoSalida])
async def leer_dispositivos(
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.es_admin:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver los dispositivos")
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

@app.delete("/usuarios/{usuario_id}")
async def eliminar_usuario(
    usuario_id: uuid.UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.es_admin:
        raise HTTPException(status_code=403, detail="No tienes permiso para eliminar usuarios")
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

@app.put("/usuarios/{usuario_id}/restablecer_contrasena", response_model=UsuarioSalida)
async def restablecer_contrasena(
    usuario_id: uuid.UUID,
    nueva_contrasena: NuevaContrasena,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.es_admin:
        logger.warning(f"Usuario no autorizado {current_user.email} intentó restablecer una contraseña")
        raise HTTPException(status_code=403, detail="No tienes permiso para restablecer contraseñas")
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

@app.delete("/notificaciones/{notificacion_id}")
async def eliminar_notificacion(
    notificacion_id: uuid.UUID,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.es_admin:
        raise HTTPException(status_code=403, detail="No tienes permiso para eliminar notificaciones")
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

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, token: str = Query(...)):
    db = SessionLocal()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        user = db.query(Usuario).filter(Usuario.email == email, Usuario.soft_delete == False).first()
        if user is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        await manager.connect(websocket, client_id, str(user.id))
        try:
            while True:
                data = await websocket.receive_text()
                # Procesar mensajes recibidos del cliente si es necesario
        except WebSocketDisconnect:
            manager.disconnect(client_id)
    except jwt.PyJWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    except Exception as e:
        logger.error(f"Error en la conexión WebSocket para el cliente {client_id}: {str(e)}")
        manager.disconnect(client_id)
    finally:
        db.close()

# Funciones auxiliares
async def enviar_notificaciones(notificacion_id: str):
    with get_db_transaction() as db:
        try:
            notificacion = db.query(Notificacion).options(
                joinedload(Notificacion.notificaciones_dispositivo).joinedload(NotificacionDispositivo.dispositivo)
            ).filter(Notificacion.id == uuid.UUID(notificacion_id), Notificacion.soft_delete == False).first()

            if not notificacion:
                logger.warning(f"Notificación {notificacion_id} no encontrada al intentar enviarla")
                return

            total_dispositivos = len(notificacion.notificaciones_dispositivo)
            dispositivos_online = 0
            notificaciones_enviadas = 0
            notificaciones_fallidas = 0

            for notif_dispositivo in notificacion.notificaciones_dispositivo:
                if notif_dispositivo.dispositivo.esta_online:
                    dispositivos_online += 1
                    usuario = db.query(Usuario).filter(Usuario.id == notif_dispositivo.dispositivo.usuario_id).first()
                    configuracion_sonido = db.query(ConfiguracionSonido).filter(ConfiguracionSonido.usuario_id == usuario.id).first()
                    sonido = configuracion_sonido.sonido if configuracion_sonido else "default.mp3"

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
                        await manager.send_personal_message(mensaje, str(notif_dispositivo.dispositivo.id))
                        notif_dispositivo.enviada = True
                        notif_dispositivo.fecha_envio = datetime.now(timezone.utc)
                        notificaciones_enviadas += 1
                        logger.info(f"Notificación {notificacion_id} enviada al dispositivo {notif_dispositivo.dispositivo.id} del usuario {usuario.email}")
                    except Exception as e:
                        notificaciones_fallidas += 1
                        logger.error(f"Error al enviar notificación {notificacion_id} al dispositivo {notif_dispositivo.dispositivo.id} del usuario {usuario.email}: {str(e)}")

            db.commit()
            logger.info(f"Resumen de envío para la notificación {notificacion_id}:")
            logger.info(f"- Total de dispositivos: {total_dispositivos}")
            logger.info(f"- Dispositivos en línea: {dispositivos_online}")
            logger.info(f"- Notificaciones enviadas con éxito: {notificaciones_enviadas}")
            logger.info(f"- Notificaciones fallidas: {notificaciones_fallidas}")
        except Exception as e:
            logger.error(f"Error inesperado al enviar notificaciones: {str(e)}")
            raise

async def enviar_notificaciones_pendientes(dispositivo_id: str):
    with get_db_transaction() as db:
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

def verificar_dispositivos_inactivos():
    with get_db_transaction() as db:
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
                client_id = manager.user_connections.get(str(dispositivo.usuario_id))
                if client_id:
                    websocket = manager.active_connections.get(client_id)
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
    
if __name__ == "__main__":
    import uvicorn
    APP_HOST = os.getenv("APP_HOST")
    APP_PORT = int(os.getenv("APP_PORT"))
    
    # Verificar y corregir asociaciones al inicio
    with SessionLocal() as db:
        verificar_y_corregir_asociaciones(db)
    
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
