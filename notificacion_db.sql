CREATE DATABASE notificacion_app;
\c notificacion_app
CREATE SCHEMA api;
SET search_path TO api;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE usuarios (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	nombre VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL,
    es_admin BOOLEAN DEFAULT FALSE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    soft_delete BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_usuarios_email ON usuarios(email);z

CREATE TABLE dispositivos (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    usuario_id UUID REFERENCES usuarios(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    esta_online BOOLEAN DEFAULT FALSE,
    ultimo_acceso TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modelo VARCHAR(100),
    sistema_operativo VARCHAR(100),
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    soft_delete BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_dispositivos_token ON dispositivos(token);
CREATE INDEX idx_dispositivos_usuario_id ON dispositivos(usuario_id);

CREATE TABLE notificaciones (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    titulo VARCHAR(255) NOT NULL,
    mensaje TEXT NOT NULL,
    imagen_url TEXT,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    soft_delete BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_notificaciones_fecha_creacion ON notificaciones(fecha_creacion);

CREATE TABLE notificaciones_dispositivo (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    notificacion_id UUID REFERENCES notificaciones(id) ON DELETE CASCADE,
    dispositivo_id UUID REFERENCES dispositivos(id) ON DELETE CASCADE,
    enviada BOOLEAN DEFAULT FALSE,
    leida BOOLEAN DEFAULT FALSE,
    fecha_envio TIMESTAMP,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    soft_delete BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_notificaciones_dispositivo_notificacion_id ON notificaciones_dispositivo(notificacion_id);
CREATE INDEX idx_notificaciones_dispositivo_dispositivo_id ON notificaciones_dispositivo(dispositivo_id);

CREATE TABLE configuracion_sonidos (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    usuario_id UUID REFERENCES usuarios(id) ON DELETE CASCADE,
    sonido VARCHAR(100) DEFAULT 'default.mp3',
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    soft_delete BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_configuracion_sonidos_usuario_id ON configuracion_sonidos(usuario_id);

CREATE TABLE refresh_tokens (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    usuario_id UUID REFERENCES usuarios(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    fecha_expiracion TIMESTAMP NOT NULL,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    soft_delete BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_refresh_tokens_token ON refresh_tokens(token);
CREATE INDEX idx_refresh_tokens_usuario_id ON refresh_tokens(usuario_id);

INSERT INTO usuarios (nombre, email, password, es_admin) 
VALUES ('Admin', 'admin@empresa.com', 'contraseña_segura', TRUE);
INSERT INTO usuarios (nombre, email, password) 
VALUES ('Usuario', 'usuario@empresa.com', 'contraseña_usuario');

-- Nuevas funciones
-- Nuevas funciones
-- Nuevas funciones
-- Nuevas funciones

-- Añadir el campo 'sonando' a la tabla notificaciones_dispositivo
ALTER TABLE notificaciones_dispositivo
ADD COLUMN sonando BOOLEAN DEFAULT FALSE;

-- Indice para notificaciones no leídas
CREATE INDEX idx_notificaciones_dispositivo_leida ON notificaciones_dispositivo(leida);

-- Indice para dispositivos en línea
CREATE INDEX idx_dispositivos_esta_online ON dispositivos(esta_online);

-- Añadir un campo para almacenar los dispositivos objetivo en la tabla notificaciones
ALTER TABLE notificaciones
ADD COLUMN dispositivos_objetivo UUID[] DEFAULT NULL;

-- nuevos cambios 10/23/2024
-- nuevos cambios 10/23/2024
-- nuevos cambios 10/23/2024
-- nuevos cambios 10/23/2024
-- nuevos cambios 10/23/2024

-- Primero, eliminar índices y restricciones existentes
DROP INDEX IF EXISTS api.idx_dispositivos_token;
ALTER TABLE api.dispositivos DROP CONSTRAINT IF EXISTS dispositivos_token_key;

-- Modificar la tabla dispositivos
ALTER TABLE api.dispositivos
    DROP COLUMN IF EXISTS token,
    ADD COLUMN IF NOT EXISTS session_id VARCHAR(255) UNIQUE,
    ADD COLUMN IF NOT EXISTS device_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS device_name VARCHAR(255);

-- Crear nuevos índices
CREATE INDEX IF NOT EXISTS idx_dispositivos_session_id ON api.dispositivos(session_id);
CREATE INDEX IF NOT EXISTS idx_dispositivos_device_id ON api.dispositivos(device_id);

-- Crear índice compuesto para device_id y usuario_id (asegura dispositivo único por usuario)
CREATE UNIQUE INDEX IF NOT EXISTS idx_dispositivos_usuario_device 
ON api.dispositivos(usuario_id, device_id) 
WHERE soft_delete = false AND esta_online = true;

-- Crear función para limpiar sesiones antiguas
CREATE OR REPLACE FUNCTION api.limpiar_sesiones_inactivas()
RETURNS void AS $$
BEGIN
    UPDATE api.dispositivos
    SET esta_online = false,
        session_id = NULL,
        fecha_actualizacion = CURRENT_TIMESTAMP
    WHERE ultimo_acceso < NOW() - INTERVAL '30 minutes'
    AND esta_online = true;
END;
$$ LANGUAGE plpgsql;

-- Crear función para cerrar sesiones anteriores
CREATE OR REPLACE FUNCTION api.cerrar_sesiones_anteriores(p_usuario_id UUID)
RETURNS void AS $$
BEGIN
    UPDATE api.dispositivos
    SET esta_online = false,
        session_id = NULL,
        fecha_actualizacion = CURRENT_TIMESTAMP
    WHERE usuario_id = p_usuario_id 
    AND esta_online = true;
END;
$$ LANGUAGE plpgsql;

-- Función para limpiar dispositivos antiguos
CREATE OR REPLACE FUNCTION api.limpiar_dispositivos_inactivos()
RETURNS void AS $$
BEGIN
    UPDATE api.dispositivos
    SET soft_delete = true,
        fecha_actualizacion = CURRENT_TIMESTAMP
    WHERE ultimo_acceso < NOW() - INTERVAL '30 days'
    AND esta_online = false;
END;
$$ LANGUAGE plpgsql;

-- no apply
-- Crear trabajo programado para limpiar sesiones y dispositivos 
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Programar limpieza de sesiones cada 15 minutos
SELECT cron.schedule('*/15 * * * *', $$SELECT api.limpiar_sesiones_inactivas()$$);

-- Programar limpieza de dispositivos diariamente
SELECT cron.schedule('0 0 * * *', $$SELECT api.limpiar_dispositivos_inactivos()$$);

-- APPLY
-- Modificar los triggers existentes o crear nuevos si es necesario
CREATE OR REPLACE FUNCTION api.actualizar_fecha_modificacion()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fecha_actualizacion = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ERROR
-- Asegurar que los triggers estén actualizados
DROP TRIGGER IF EXISTS actualizar_fecha_dispositivos ON api.dispositivos;
CREATE TRIGGER actualizar_fecha_dispositivos
    BEFORE UPDATE ON api.dispositivos
    FOR EACH ROW
    EXECUTE FUNCTION api.actualizar_fecha_modificacion();

-- no appply
-- Comentarios para documentación 
COMMENT ON TABLE api.dispositivos IS 'Almacena información de dispositivos y sesiones de usuarios';
COMMENT ON COLUMN api.dispositivos.session_id IS 'Identificador único de sesión activa';
COMMENT ON COLUMN api.dispositivos.device_id IS 'Identificador único del dispositivo físico';
COMMENT ON COLUMN api.dispositivos.device_name IS 'Nombre amigable del dispositivo';

-- Agregar algunos índices adicionales para optimizar consultas comunes
CREATE INDEX IF NOT EXISTS idx_dispositivos_estado_sesion 
ON api.dispositivos(esta_online, session_id) 
WHERE soft_delete = false;

CREATE INDEX IF NOT EXISTS idx_dispositivos_ultimo_acceso 
ON api.dispositivos(ultimo_acceso) 
WHERE esta_online = true AND soft_delete = false;







-- uso posterior
-- Función para actualizar automáticamente las fechas de actualización
CREATE OR REPLACE FUNCTION update_fecha_actualizacion()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fecha_actualizacion = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- NO APLICADAS AUN
-- Aplicar la función de actualización a todas las tablas relevantes
CREATE TRIGGER update_usuarios_fecha_actualizacion
BEFORE UPDATE ON usuarios
FOR EACH ROW EXECUTE FUNCTION update_fecha_actualizacion();

CREATE TRIGGER update_dispositivos_fecha_actualizacion
BEFORE UPDATE ON dispositivos
FOR EACH ROW EXECUTE FUNCTION update_fecha_actualizacion();

CREATE TRIGGER update_notificaciones_fecha_actualizacion
BEFORE UPDATE ON notificaciones
FOR EACH ROW EXECUTE FUNCTION update_fecha_actualizacion();

CREATE TRIGGER update_notificaciones_dispositivo_fecha_actualizacion
BEFORE UPDATE ON notificaciones_dispositivo
FOR EACH ROW EXECUTE FUNCTION update_fecha_actualizacion();

CREATE TRIGGER update_configuracion_sonidos_fecha_actualizacion
BEFORE UPDATE ON configuracion_sonidos
FOR EACH ROW EXECUTE FUNCTION update_fecha_actualizacion();

CREATE TRIGGER update_refresh_tokens_fecha_actualizacion
BEFORE UPDATE ON refresh_tokens
FOR EACH ROW EXECUTE FUNCTION update_fecha_actualizacion();

CREATE TRIGGER update_estadisticas_notificaciones_fecha_actualizacion
BEFORE UPDATE ON estadisticas_notificaciones
FOR EACH ROW EXECUTE FUNCTION update_fecha_actualizacion();
