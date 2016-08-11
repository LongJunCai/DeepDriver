package deepDriver.dl.aml.distribution;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

public class P2PServer {
	ServerSocket server = null;
	public static String KEY_SRV_PORT = "KEY_SRV_PORT";
	public static String KEY_SRV_HOST = "KEY_SRV_HOST";
	static int sport = 11212;
	static String srvHost = "127.0.0.1"; 
	List<ClientVo> clients = new ArrayList<ClientVo>();
	public void setup(int clients) {
		try {
			int port = DistributionEnvCfg.getCfg().getInt(KEY_SRV_PORT);
			if (port > 0) {
				sport = port;
			}
			server = new ServerSocket(sport); 
			System.out.println("Server is started on "+sport);
			System.out.println(clients+" clients need to be servered");
			for (int i = 0; i < clients; i++) {
				System.out.println("try to connect to "+i+" clients");
				buildConnection();
			}			
		} catch (Exception e) {
			
		}
		
	}
	
	
	
		
	public static String ObjCommand = "-c sendObj";
	public static String CollectObj = "-c collectObj";
	
	public static String OK = "OK";
	public Object [] collectObjs() {
//		distributeCommand(CollectObj);
		Object [] objs = new Object[clients.size()];
		for (int i = 0; i < objs.length; i++) {
			ClientVo cv = clients.get(i);
			try {
				objs[i] = cv.getOis().readUnshared();
				response(cv, i, OK);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return objs;
	}
	
	static boolean debug = false;
	public static void debug(String msg) {
		if (debug) {
			System.out.println(msg);
		}
	}
	
	public static void info(String msg) {
		System.out.println(msg);
	}
	
	public void response(ClientVo cv, int i, String msg) throws IOException {
		if (noFutherCom) {
			return;
		}
		cv.oos.writeUnshared(msg);	
		cv.oos.flush();
		debug("Response to client"+i+", with "+msg);
	}
	
	boolean noFutherCom = false;
	
	public boolean getResponse(ClientVo cv, int i) throws Exception {
		if (noFutherCom) {
			return false;
		}
		String line = (String) cv.getOis().readUnshared();
		if (OK.equals(line.trim())) {
			debug("Send to client "+i+" already.");
			return true;
		}
		cv.getOos().reset();
		debug("Send to client "+i+" already, and response "+line );
		return false;
	}
	
	public void distributeCommand(String command) throws Exception {
//		info("start to send command "+command);
		for (int i = 0; i < clients.size(); i++) {
			ClientVo cv = clients.get(i);
			cv.getOos().writeUnshared(command);
			cv.getOos().flush();
			getResponse(cv, i);
		}
		debug("finished.");
	}
	
	public void distributeObjects(Object [] objs) throws Exception {
//		distributeCommand(ObjCommand);
		for (int i = 0; i < clients.size(); i++) {
			ClientVo cv = clients.get(i);
			try {
				cv.getOos().writeUnshared(objs[i]);
				cv.getOos().flush();
				cv.getOos().reset();
				getResponse(cv, i);				
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void distributeObject(Object obj) throws Exception {
//		distributeCommand(ObjCommand);
		for (int i = 0; i < clients.size(); i++) {
			ClientVo cv = clients.get(i);
			try {  
				cv.getOos().writeUnshared(obj);
//				cv.getOos().writeObject(obj);
				cv.getOos().flush();
				cv.getOos().reset();
				getResponse(cv, i);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void rebuildConnection(ClientVo cv) throws Exception {
		cv.oos.close();
		cv.ois.close();
		
	}
	
	public void close() throws Exception {
		for (int i = 0; i < clients.size(); i++) {
			ClientVo cv = clients.get(i);
			try {
				cv.oos.close();
				cv.ois.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}		
	}
	
	public void close(ClientVo cv) throws Exception {
		cv.oos.close();
		cv.ois.close();
	}
	
	public void buildConnection() {
		try {			
			Socket socket = server.accept();
//			String line;
//			BufferedReader is = null;
//			PrintWriter os = null;
//			is = new BufferedReader(new InputStreamReader(
//					socket.getInputStream()));
//			os = new PrintWriter(socket.getOutputStream());
//			BufferedReader sin = new BufferedReader(new InputStreamReader(
//					System.in));
			System.out.println(socket.getSoTimeout());
			socket.setKeepAlive(true);
			socket.setSoTimeout(1000 * 60 * 60);
			clients.add(new ClientVo(socket));
			info("Client:" + clients.size()+" is in.");
//			line = sin.readLine();
//			while (!line.equals("bye")) {
//				os.println(line);
//				os.flush();
//				System.out.println("Server:" + line);
//				System.out.println("Client:" + is.readLine());
//				line = sin.readLine();
//			} 
		} catch (Exception e) {
			System.out.println("Error:" + e);
		}
	}
	
	
	public List<ClientVo> getClients() {
		return clients;
	}



	public void setClients(List<ClientVo> clients) {
		this.clients = clients;
	}



	public static void main(String args[]) {

	}

	public void collectState() throws Exception { 
		for (int i = 0; i < clients.size(); i++) {
			ClientVo cv = clients.get(i); 
			getResponse(cv, i);
			debug("Client "+i+" is ready.");
		}
		
	}

}
