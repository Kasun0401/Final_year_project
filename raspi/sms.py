import serial
import time

ser = serial.Serial("/dev/ttyS0", 9600, timeout=1)

def send_sms(number, message):
    ser.write('AT+CMGF=1\r'.encode())
    time.sleep(1)
    ser.write('AT+CMGS="{}"\r'.format(number).encode())
    time.sleep(1)
    ser.write(message.encode())
    time.sleep(1)
    ser.write(chr(26).encode())  # Ctrl-Z to send the message

if __name__ == "__main__":
    number = "0718684441"
    message = "i AM KASUN!"

    send_sms(number, message)
