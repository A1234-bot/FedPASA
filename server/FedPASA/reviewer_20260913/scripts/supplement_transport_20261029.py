"""Framed TCP loopback transport, no shared-memory tensor reducers."""
import os, pickle, socket, struct, time

def receive_exact(sock, count):
    data=bytearray(count);view=memoryview(data);offset=0
    while offset<count:
        n=sock.recv_into(view[offset:])
        if not n:raise EOFError('peer disconnected')
        offset+=n
    return data

class Wire:
    def __init__(self,sock,client):
        self.sock=sock;self.client=client;self.rows=[]
    def send(self,stage,obj):
        t=time.perf_counter();raw=pickle.dumps((stage,obj),protocol=5);encode=time.perf_counter()-t
        t=time.perf_counter()
        mbps=float(os.environ.get('SUPPLEMENT_MBIT','0'))
        delay=float(os.environ.get('SUPPLEMENT_RTT_MS','0'))/2000
        if delay:time.sleep(delay)
        framed=struct.pack('!Q',len(raw))+raw
        if mbps>0:
            # A 64 KiB token-bucket-like sender pace; sleep before releasing each chunk.
            began=time.perf_counter()
            for offset in range(0,len(framed),65536):
                end=min(offset+65536,len(framed))
                wait=end/(mbps*1e6/8)-(time.perf_counter()-began)
                if wait>0:time.sleep(wait)
                self.sock.sendall(framed[offset:end])
        else:self.sock.sendall(framed)
        self.rows.append(dict(direction='send',stage=stage,client=self.client,payload_bytes=len(raw),framed_bytes=len(raw)+8,serialize_seconds=encode,io_seconds=time.perf_counter()-t))
    def recv(self,expected=None):
        t=time.perf_counter();size=struct.unpack('!Q',receive_exact(self.sock,8))[0]
        if size>2**31:raise ValueError('invalid frame size')
        raw=receive_exact(self.sock,size);io=time.perf_counter()-t;t=time.perf_counter();stage,obj=pickle.loads(raw);decode=time.perf_counter()-t
        if expected is not None:assert stage==expected,(stage,expected)
        self.rows.append(dict(direction='recv',stage=stage,client=self.client,payload_bytes=size,framed_bytes=size+8,deserialize_seconds=decode,io_seconds=io))
        return obj
    def close(self):self.sock.close()

def listener():
    s=socket.socket();s.bind(('127.0.0.1',0));s.listen(10);s.settimeout(900);return s

def connect(port,client):
    s=socket.create_connection(('127.0.0.1',port),timeout=900);s.settimeout(7200);return Wire(s,client)

def wire_summary(wires):
    rows=[r for w in wires for r in w.rows]
    return dict(uplink_payload_bytes=sum(r['payload_bytes'] for r in rows if r['direction']=='recv'),downlink_payload_bytes=sum(r['payload_bytes'] for r in rows if r['direction']=='send'),uplink_framed_bytes=sum(r['framed_bytes'] for r in rows if r['direction']=='recv'),downlink_framed_bytes=sum(r['framed_bytes'] for r in rows if r['direction']=='send'),uplink_stages=len({r['stage'] for r in rows if r['direction']=='recv'}),downlink_stages=len({r['stage'] for r in rows if r['direction']=='send'}),server_serialize_seconds=sum(r.get('serialize_seconds',0) for r in rows),server_deserialize_seconds=sum(r.get('deserialize_seconds',0) for r in rows),messages=rows,transport='Application-paced TCP loopback when SUPPLEMENT_MBIT/RTT_MS set; pickle5 payload plus 8-byte application frame header; excludes TCP/IP and kernel retransmission bytes')
