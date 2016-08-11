package deepDriver.dl.aml.distribution;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

public class P2PBase {

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
		}
		return null;
	}
	
	public String receiveCommand() throws ClassNotFoundException {
		try {
			String cmd = (String) ois.readUnshared();
			response(P2PServer.OK);
			return cmd;
		} catch (IOException e) { 
			e.printStackTrace();
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
		}
	}
	

}
