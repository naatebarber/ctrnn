# CTRNN

CTRNN framework for Rust.

Network types:

 - `HashNetwork`: Network with flexible HashMap neuron connections, ideal for large, sparse networks.
 - `SsmNetwork`: Single-threaded network using a centralized synaptic strength matrix.
 - `PowerNetwork`: Multi-threaded, distributable network using a centralized synaptic strength matrix.

Usage:

```rust
use ctrnn::PowerNetwork;

// PowerNetwork(size, d_in, d_out, worker_cores)
let mut power = PowerNetwork::new(24, 1, 1, 8).unwrap();

// PowerNetwork.weave(neural_density)
power.weave(0.8)

// PowerNetwork.forward(inputs, next_tau, step_size)
let inputs = vec![0.3];
let next_tau = ctrnn::get_ts() + 3.;
let step_size = 0.001;

power.forward(inputs, next_tau, step_size);

let losses: Vec<f64> = my_loss_fn();
let learning_rate = 0.01;

power.backward(learning_rate, losses).unwrap();
```