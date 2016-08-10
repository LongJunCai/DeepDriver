package com.test;

import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JTextField;

public class TestJFrame extends JFrame {
	
	private static final long serialVersionUID = 1L;

	public void test() {
		final JTextField ft = new JTextField();
		JButton jb = new JButton();
		Container cc = getContentPane();
		cc.setLayout(new FlowLayout());
		ft.setText("Hello World");
		jb.setText("Clean");
		this.getContentPane().add(ft);
		this.getContentPane().add(jb);
		
		jb.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				ft.setText("");				
			}
		});
		
		ft.addKeyListener(new KeyListener() {
			
			@Override
			public void keyTyped(KeyEvent kee) {
				char ch = kee.getKeyChar();
				System.out.println(ch);
			}
			
			@Override
			public void keyReleased(KeyEvent e) {
				
			}
			
			@Override
			public void keyPressed(KeyEvent e) {
				
			}
		});
	}
	
	public static void main(String[] args) {
		TestJFrame tj = new TestJFrame();		

		tj.setSize(200, 300);
		tj.setVisible(true);
		tj.test();
	}
}
