using System;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.IO;

class Program
{
    static readonly object ConsoleLock = new object();
    static Socket TCPSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
    static Socket UDPSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
    static Thread AnnoucerThread;
    static List<Thread> WorkerThreads = new List<Thread>();
    static void Announcer()
    {
        byte[] message = Encoding.ASCII.GetBytes("EmoRecog: " + ((IPEndPoint)TCPSocket.LocalEndPoint).Port);
        EndPoint endPoint = new IPEndPoint(IPAddress.Broadcast, 6969);
        while (true)
        {
            try
            {
                UDPSocket.SendTo(message, endPoint);
            }
            catch (Exception e)
            {
                lock (ConsoleLock)
                {
                    Console.WriteLine("** Unable to brodcast, reason = " + e.Message);
                    return;
                }
            }
            Thread.Sleep(1000);
        }
    }
    static void Worker(object o)
    {
        Socket s = (Socket)o;
        string remoteip = ((IPEndPoint)s.RemoteEndPoint).Address.ToString();
        while (s.Connected)
        {
            byte[] message = new byte[1024];
            int n = 0;
            try
            {
                n = s.Receive(message, 1, SocketFlags.None);
            }
            catch (Exception)
            {
                break;
            }
            bool Terminate = false;
            switch (message[0])
            {
                case 1:
                    {
                        s.Receive(message, 4, SocketFlags.None);
                        int PhotoSize = BitConverter.ToInt32(message, 0);
                        message = new byte[PhotoSize];
                        List<byte> Photo = new List<byte>(PhotoSize);
                        int i = 0;
                        while (i < PhotoSize)
                        {
                            int SegmentSize = s.Receive(message);
                            if (SegmentSize == 0)
                            {
                                Terminate = true;
                                break;
                            }
                            for (int j = 0; j < SegmentSize; j++)
                            {
                                Photo.Add(message[j]);
                            }
                            i += SegmentSize;
                        }
                        lock (ConsoleLock)
                        {
                            Console.WriteLine(remoteip + ": received a " + PhotoSize + " bytes photo");
                        }
                        File.WriteAllBytes("image.jpg", Photo.ToArray());
                        //Image processing goes here
                    }
                    break;
                default:
                    Terminate = true;
                    lock (ConsoleLock)
                    {
                        Console.WriteLine(remoteip + ": received an unknown type id " + message[0]);
                    }
                    break;
            }
            if (Terminate)
            {
                break;
            }
        }
        lock (ConsoleLock)
        {
            Console.WriteLine("## Terminating connection with " + remoteip);
        }
        s.Shutdown(SocketShutdown.Both);
        s.Close();
        lock (WorkerThreads)
        {
            WorkerThreads.Remove(Thread.CurrentThread);
        }
    }
    static void Main(string[] args)
    {
        Console.WriteLine("## Initializing TCP Socket");
        try
        {
            TCPSocket.Bind(new IPEndPoint(IPAddress.Any, 0));
            TCPSocket.Listen(100);
        }
        catch (Exception e)
        {
            Console.WriteLine("** Failed to initalize TCP Socket, Reason : " + e.Message);
            return;
        }
        Console.WriteLine("-> TCP Socket initalized successfully, listening to 100 connections and is bound to port "
                          + ((IPEndPoint)TCPSocket.LocalEndPoint).Port);
        Console.WriteLine("## Enabling Broadcast in UDP Socket");
        UDPSocket.EnableBroadcast = true;
        Console.WriteLine("## Creating announce thread");
        AnnoucerThread = new Thread(new ThreadStart(Announcer));
        AnnoucerThread.Start();
        Console.WriteLine("## Waiting for connections");
        while (true)
        {
            Socket s = TCPSocket.Accept();
            lock (ConsoleLock)
            {
                Console.WriteLine("-> " + ((IPEndPoint)s.RemoteEndPoint).Address.ToString() + " is connected");
            }
            Thread t = new Thread(new ParameterizedThreadStart(Worker));
            t.Start(s);
            lock (WorkerThreads)
            {
                WorkerThreads.Add(t);
            }
        }
    }
}
