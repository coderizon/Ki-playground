let calliopeDevice;
let calliopeServer;
let calliopeUartService;
let calliopeUartCharacteristic;
let connectionListener = () => {};

// Calliope mini (Rev2) nutzt wie der micro:bit den Nordic UART Service
const CALLIOPE_UART_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
const CALLIOPE_UART_WRITE_CHARACTERISTIC_CANDIDATES = [
  '6e400003-b5a3-f393-e0a9-e50e24dcca9e', // TX laut Calliope-Doku
  '6e400002-b5a3-f393-e0a9-e50e24dcca9e', // Fallback, falls Schreiben auf RX erwartet wird
];
const CALLIOPE_NAME_PREFIXES = ['Calliope mini', 'CALLIOPE mini', 'CALLIOPE MINI'];

function buildCalliopeFilters() {
  const filters = CALLIOPE_NAME_PREFIXES.map((prefix) => ({ namePrefix: prefix }));
  filters.push({ services: [CALLIOPE_UART_SERVICE_UUID] });
  return filters;
}

export function setCalliopeConnectionListener(listener) {
  connectionListener = typeof listener === 'function' ? listener : () => {};
}

export function isCalliopeConnected() {
  return Boolean(
    calliopeDevice &&
    calliopeDevice.gatt &&
    calliopeDevice.gatt.connected &&
    calliopeUartCharacteristic
  );
}

export async function connectCalliope() {
  if (!navigator.bluetooth) {
    alert('Web Bluetooth wird von diesem Browser nicht unterstützt.');
    return;
  }

  try {
    calliopeDevice = await navigator.bluetooth.requestDevice({
      filters: buildCalliopeFilters(),
      optionalServices: [CALLIOPE_UART_SERVICE_UUID],
    });

    calliopeServer = await calliopeDevice.gatt.connect();
    calliopeUartService = await calliopeServer.getPrimaryService(CALLIOPE_UART_SERVICE_UUID);
    calliopeUartCharacteristic = await findCalliopeCharacteristic(calliopeUartService);

    console.log('✅ Calliope mini verbunden');
    alert('Calliope mini erfolgreich verbunden!');
    connectionListener(true);
    calliopeDevice.addEventListener('gattserverdisconnected', handleCalliopeDisconnect);
  } catch (error) {
    console.error('❌ Fehler beim Verbinden mit dem Calliope mini:', error);
    alert('Calliope mini konnte nicht verbunden werden.');
    cleanupCalliopeState();
    connectionListener(false);
    throw error;
  }
}

function handleCalliopeDisconnect() {
  console.warn('⚠️ Calliope mini Verbindung getrennt.');
  alert('Calliope mini Verbindung getrennt.');
  cleanupCalliopeState();
  connectionListener(false);
}

function cleanupCalliopeState() {
  calliopeUartCharacteristic = null;
  calliopeUartService = null;
  if (calliopeServer) {
    try {
      calliopeServer.disconnect();
    } catch (_) {
      // already disconnected
    }
  }
  calliopeServer = null;
  calliopeDevice = null;
}

async function findCalliopeCharacteristic(service) {
  for (const uuid of CALLIOPE_UART_WRITE_CHARACTERISTIC_CANDIDATES) {
    try {
      return await service.getCharacteristic(uuid);
    } catch (_) {
      // continue searching
    }
  }
  throw new Error('Kein kompatibles UART-Charakteristikum im Calliope mini Service gefunden.');
}

export async function sendToCalliope(text) {
  if (!calliopeUartCharacteristic) {
    console.warn('Calliope mini ist nicht verbunden – Nachricht wurde nicht gesendet.');
    return;
  }

  try {
    const data = new TextEncoder().encode(String(text) + '\n');
    if (typeof calliopeUartCharacteristic.writeValueWithoutResponse === 'function') {
      await calliopeUartCharacteristic.writeValueWithoutResponse(data);
    } else {
      await calliopeUartCharacteristic.writeValue(data);
    }
  } catch (error) {
    console.error('Senden zum Calliope mini fehlgeschlagen:', error);
  }
}
