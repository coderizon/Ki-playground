let arduinoDevice;
let arduinoServer;
let arduinoUartService;
let arduinoUartCharacteristic;
let connectionListener = () => {};

// Der UNO R4 (WiFi) kann mithilfe der ArduinoBLE-Library den weit verbreiteten Nordic UART Service bereitstellen.
// Passe die UUIDs an deine Sketch-Konfiguration an, falls du andere Werte verwendest.
const ARDUINO_UART_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
const ARDUINO_UART_WRITE_CHARACTERISTIC_CANDIDATES = [
  '6e400002-b5a3-f393-e0a9-e50e24dcca9e', // Standard-Nordic RX (Central → Peripheral)
  '6e400003-b5a3-f393-e0a9-e50e24dcca9e', // Fallback für Boards, die auf TX schreiben lassen
];
const ARDUINO_NAME_PREFIXES = ['Arduino UNO R4', 'UNO R4', 'Arduino UNO'];

function buildArduinoFilters() {
  const filters = ARDUINO_NAME_PREFIXES.map((prefix) => ({
    namePrefix: prefix,
    services: [ARDUINO_UART_SERVICE_UUID],
  }));
  filters.push({ services: [ARDUINO_UART_SERVICE_UUID] });
  return filters;
}

export function setArduinoConnectionListener(listener) {
  connectionListener = typeof listener === 'function' ? listener : () => {};
}

export function isArduinoConnected() {
  return Boolean(
    arduinoDevice &&
    arduinoDevice.gatt &&
    arduinoDevice.gatt.connected &&
    arduinoUartCharacteristic
  );
}

export async function connectArduino() {
  if (!navigator.bluetooth) {
    alert('Web Bluetooth wird von diesem Browser nicht unterstützt.');
    return;
  }

  try {
    arduinoDevice = await navigator.bluetooth.requestDevice({
      filters: buildArduinoFilters(),
      optionalServices: [ARDUINO_UART_SERVICE_UUID],
    });

    arduinoServer = await arduinoDevice.gatt.connect();
    arduinoUartService = await arduinoServer.getPrimaryService(ARDUINO_UART_SERVICE_UUID);
    arduinoUartCharacteristic = await findArduinoCharacteristic(arduinoUartService);

    console.log('✅ Arduino UNO R4 verbunden');
    alert('Arduino UNO R4 erfolgreich verbunden!');
    connectionListener(true);
    arduinoDevice.addEventListener('gattserverdisconnected', handleArduinoDisconnect);
  } catch (error) {
    console.error('❌ Fehler beim Verbinden mit dem Arduino UNO R4:', error);
    alert('Arduino UNO R4 konnte nicht verbunden werden. Bitte prüfe den Bluetooth-Status und das laufende Sketch.');
    cleanupArduinoState();
    connectionListener(false);
    throw error;
  }
}

function handleArduinoDisconnect() {
  console.warn('⚠️ Arduino UNO R4 Verbindung getrennt.');
  alert('Arduino UNO R4 Verbindung getrennt.');
  cleanupArduinoState();
  connectionListener(false);
}

function cleanupArduinoState() {
  arduinoUartCharacteristic = null;
  arduinoUartService = null;
  if (arduinoServer) {
    try {
      arduinoServer.disconnect();
    } catch (_) {
      // already disconnected
    }
  }
  arduinoServer = null;
  arduinoDevice = null;
}

async function findArduinoCharacteristic(service) {
  for (const uuid of ARDUINO_UART_WRITE_CHARACTERISTIC_CANDIDATES) {
    try {
      return await service.getCharacteristic(uuid);
    } catch (_) {
      // continue searching
    }
  }
  throw new Error('Kein kompatibles UART-Charakteristikum im UNO R4 Service gefunden.');
}

export async function sendToArduino(text) {
  if (!arduinoUartCharacteristic) {
    console.warn('Arduino UNO R4 ist nicht verbunden – Nachricht wurde nicht gesendet.');
    return;
  }

  try {
    const data = new TextEncoder().encode(String(text) + '\n');
    if (typeof arduinoUartCharacteristic.writeValueWithoutResponse === 'function') {
      await arduinoUartCharacteristic.writeValueWithoutResponse(data);
    } else {
      await arduinoUartCharacteristic.writeValue(data);
    }
  } catch (error) {
    console.error('Senden zum Arduino UNO R4 fehlgeschlagen:', error);
  }
}
