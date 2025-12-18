let microbitDevice;
let microbitServer;
let microbitUartService;
let microbitUartCharacteristic;
let connectionListener = () => {};

// Nordic UART (micro:bit)
const MICROBIT_UART_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
const MICROBIT_UART_WRITE_CHARACTERISTIC_CANDIDATES = [
  '6e400003-b5a3-f393-e0a9-e50e24dcca9e', // Browser → micro:bit (TX)
  '6e400002-b5a3-f393-e0a9-e50e24dcca9e', // Fallback, falls RX erwartet wird
];
const MICROBIT_NAME_PREFIXES = ['BBC micro:bit', 'micro:bit'];

function buildMicrobitFilters() {
  const filters = MICROBIT_NAME_PREFIXES.map((prefix) => ({ namePrefix: prefix }));
  filters.push({ services: [MICROBIT_UART_SERVICE_UUID] });
  return filters;
}

export function setMicrobitConnectionListener(listener) {
  connectionListener = typeof listener === 'function' ? listener : () => {};
}

export function isMicrobitConnected() {
  return Boolean(
    microbitDevice &&
    microbitDevice.gatt &&
    microbitDevice.gatt.connected &&
    microbitUartCharacteristic
  );
}

export async function connectMicrobit() {
  if (!navigator.bluetooth) {
    alert('Web Bluetooth wird von diesem Browser nicht unterstützt.');
    return;
  }

  try {
    microbitDevice = await navigator.bluetooth.requestDevice({
      filters: buildMicrobitFilters(),
      optionalServices: [MICROBIT_UART_SERVICE_UUID],
    });

    microbitServer = await microbitDevice.gatt.connect();
    microbitUartService = await microbitServer.getPrimaryService(MICROBIT_UART_SERVICE_UUID);
    microbitUartCharacteristic = await findMicrobitCharacteristic(microbitUartService);

    console.log('✅ Micro:bit verbunden');
    alert('Micro:bit erfolgreich verbunden!');
    connectionListener(true);
    microbitDevice.addEventListener('gattserverdisconnected', handleMicrobitDisconnect);
  } catch (error) {
    console.error('❌ Fehler beim Verbinden mit dem Micro:bit:', error);
    alert('Micro:bit konnte nicht verbunden werden.');
    cleanupMicrobitState();
    connectionListener(false);
    throw error;
  }
}

function handleMicrobitDisconnect() {
  cleanupMicrobitState();
  connectionListener(false);
  alert('Micro:bit Verbindung getrennt.');
}

function cleanupMicrobitState() {
  microbitUartCharacteristic = null;
  microbitUartService = null;
  if (microbitServer) {
    try {
      microbitServer.disconnect();
    } catch (_) {
      // already disconnected
    }
  }
  microbitServer = null;
  microbitDevice = null;
}

async function findMicrobitCharacteristic(service) {
  for (const uuid of MICROBIT_UART_WRITE_CHARACTERISTIC_CANDIDATES) {
    try {
      return await service.getCharacteristic(uuid);
    } catch (_) {
      // continue searching
    }
  }
  throw new Error('Kein kompatibles UART-Charakteristikum im Micro:bit Service gefunden.');
}

// immer writeWithoutResponse verwenden
export async function sendToMicrobit(text) {
  if (!microbitUartCharacteristic) {
    console.warn('Micro:bit ist nicht verbunden – Nachricht wurde nicht gesendet.');
    return;
  }

  try {
    const data = new TextEncoder().encode(String(text) + '\n');
    if (typeof microbitUartCharacteristic.writeValueWithoutResponse === 'function') {
      await microbitUartCharacteristic.writeValueWithoutResponse(data);
    } else {
      await microbitUartCharacteristic.writeValue(data);
    }
  } catch (error) {
    console.error('Senden zum Micro:bit fehlgeschlagen:', error);
  }
}
