package deepDriver.dl.aml.distribution;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.net.UnknownHostException;

public class P2PBase {

	String master = "127.0.0.1";
	int sport = P2PServer.sport;
	
	Socket socket;
	ObjectOutputStream oos;
	ObjectInputStream ois;	
	
	public P2PBase(Socket socket, ObjectOutputStream oos, ObjectInputStream ois) {
		super();
		this.socket = socket;
		this.oos = oos;
		this.ois = ois;
	}
	public P2PBase() {}

	public void responseReady() throws IOException {
		response(P2PServer.OK);
	}
	
	public void response(String msg) throws IOException {
		if (noFutherCom) {
			return ;
		}
		oos.writeUnshared(msg);
//		os.println(msg);	
		oos.flush();
		oos.reset();
	}
	
	public Object receiveObj() {
		try {
			Object obj = ois.readUnshared();
//			Object obj = ois.readObject(); 
			response(P2PServer.OK);
			return obj;
		} catch (Exception e) { 
			e.printStackTrace();
			rebuild(e);
		}
		return null;
	}
	
	int retryTime = 5;
	
	public void rebuild(Exception ex) {
		try {
			ois.close();
			oos.close();
			socket.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		for (int i = 0; i < retryTime; i++) {
			try {
				socket = new Socket(master, sport);
				socket.setKeepAlive(true);
				socket.setSoTimeout(1000 * 60 * 60);
				oos = new ObjectOutputStream(socket.getOutputStream());
				ois = new ObjectInputStream(socket.getInputStream());
				break;
			} catch (IOException e) {
				e.printStackTrace();
				try {
					Thread.sleep(5 * 1000);
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
			}			
		}
		
		
	}
	
	public String receiveCommand() throws ClassNotFoundException {
		try {
			String cmd = (String) ois.readUnshared();
			response(P2PServer.OK);
			return cmd;
		} catch (IOException e) { 
			e.printStackTrace();
			rebuild(e);
		}
		return null;
	}
	
	boolean noFutherCom = false; 
	
	public boolean getResponse() throws Exception {
		if (noFutherCom) {
			return false;
		}
		String line = (String) ois.readUnshared();
		if (P2PServer.OK.equals(line.trim())) {
			P2PServer.debug("Send to server already.");
			return true;
		}
		oos.reset();
		P2PServer.debug("Send to server already, and response "+line );
		return false;
	}
	
	public void sendObj(Object obj) {
		P2PServer.debug("Prepare to send Obj");
		try {
			oos.writeUnshared(obj); 
			oos.flush();
			oos.reset();
			getResponse();
		} catch (Exception e) { 
			e.printStackTrace();
			rebuild(e);
		}
	}
	

}
