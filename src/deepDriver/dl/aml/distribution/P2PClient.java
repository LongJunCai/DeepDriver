package deepDriver.dl.aml.distribution;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

public class P2PClient extends P2PBase {
	String master = "127.0.0.1";
	int sport = P2PServer.sport;
	public void setup() {
		try {
			int port = DistributionEnvCfg.getCfg().getInt(P2PServer.KEY_SRV_PORT);
			String host = DistributionEnvCfg.getCfg().getString(P2PServer.KEY_SRV_HOST);
			if (port > 0) {
				sport = port;
			}
			if (host != null) {
				master = host;
			}
			socket=new Socket(master,sport);
			socket.setKeepAlive(true);
			socket.setSoTimeout(1000 * 60 * 60);
//			os = new PrintWriter(socket.getOutputStream());
//			is = new BufferedReader(new InputStreamReader(socket.getInputStream()));
			oos = new ObjectOutputStream(socket.getOutputStream());			
//			oos = new ObjectOutputStream(socket.getOutputStream());
//			ois = new ObjectInputStream(socket.getInputStream());
			ois = new ObjectInputStream(socket.getInputStream());
					
			System.out.println("slave setup, and connect to master");
		} catch (Exception e) {
		}
	}
	
}
