package deepDriver.dl.aml.distribution;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

public class ClientVo {
	Socket socket;
//	BufferedReader is;
//	PrintWriter os;
	ObjectOutputStream oos;
	ObjectInputStream ois;
	
	public ObjectInputStream getOis() {
		return ois;
	}

	public void setOis(ObjectInputStream ois) {
		this.ois = ois;
	}
	
	public void rebuild() {
		try {
			oos = new ObjectOutputStream(socket.getOutputStream());
//			ois = new ObjectInputStream(socket.getInputStream());
//			oos = new ObjectOutputStream(socket.getOutputStream());
			ois = new ObjectInputStream(socket.getInputStream());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public ClientVo(Socket socket) {
		super();
		this.socket = socket;
//		this.is = is;
//		this.os = os;
		try {
			oos = new ObjectOutputStream(socket.getOutputStream());
//			ois = new ObjectInputStream(socket.getInputStream());
//			oos = new ObjectOutputStream(socket.getOutputStream());
			ois = new ObjectInputStream(socket.getInputStream());
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public ObjectOutputStream getOos() {
		return oos;
	}

	public void setOos(ObjectOutputStream oos) {
		this.oos = oos;
	}

	public Socket getSocket() {
		return socket;
	}
	public void setSocket(Socket socket) {
		this.socket = socket;
	}
//	public BufferedReader getIs() {
//		return is;
//	}
//	public void setIs(BufferedReader is) {
//		this.is = is;
//	}
//	public PrintWriter getOs() {
//		return os;
//	}
//	public void setOs(PrintWriter os) {
//		this.os = os;
//	}
	
}
