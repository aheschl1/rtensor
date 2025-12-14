pub mod server;
pub mod client;
pub mod protocol;

#[cfg(test)]
mod tests {
    use std::thread;

    use crate::{backend::{remote::{client::RemoteBackend, server::RemoteServer}, Backend}, core::{primitives::TensorBase, tensor::TensorAccess, MetaTensor}};

    #[test]
    fn testing_sandbox() {
        let server_ip = "127.0.0.1";
        let server_port = 7878;
        let server_addr = format!("{}:{}", server_ip, server_port);
        println!("Server address: {}", server_addr);

        let mut server = RemoteServer::new(server_ip.parse().unwrap(), server_port);
        thread::spawn(move || {
            server.serve().unwrap();
        });
        println!("Server started, waiting for client...");
        thread::sleep(std::time::Duration::from_millis(10));

        let mut backend = RemoteBackend::new_with_address(server_ip.parse().unwrap(), server_port).unwrap();
        backend.connect().unwrap();
        let mut buffer = backend.alloc::<f32>(100).unwrap();
        println!("Allocated remote buffer: {:?}", buffer);
        let res = backend.copy_from_slice(&mut buffer, vec![1.0f32; 100].as_slice()).unwrap();
        println!("Copied data to remote buffer");
        println!("{:?}", backend);

        println!("Reading data back from remote buffer...");
        let res = backend.read(&buffer, 0).unwrap();
        println!("Read data from remote buffer: {:?}", res);

        let mut tensor = TensorBase::from_parts(
            backend, 
            buffer,
            MetaTensor::new(vec![10, 10], vec![10, 1], 0) 
        );


        println!("Created remote tensor: {:?}", tensor);

        tensor += 1.0;

        let x = tensor.get((0, 0)).unwrap();
        println!("Tensor after addition: {:?}", x);

        tensor *= 2.0;
        println!("Tensor after multiplication: {:?}", tensor.get((0, 0)).unwrap());
        
    }
}