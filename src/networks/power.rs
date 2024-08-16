use rand::{prelude::*, thread_rng, Rng};
use std::{collections::VecDeque, error::Error};

use std::{
    cmp::Ordering,
    thread::{self, JoinHandle},
    time::Duration,
};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::util::get_ts;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuronFrame {
    pub state: f64,
    pub tau: f64,
    pub loss: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BackpropFrame {
    pub step: usize,
    pub neurons: Vec<Vec<NeuronFrame>>,
    pub output_neurons: Vec<usize>,
    pub ssm: Array2<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GradientFrame {
    pub step: usize,
    pub gradient: Array2<f64>,
}

pub struct Worker {
    id: uuid::Uuid,
    pull_sock: zmq::Socket,
    push_sock: zmq::Socket,
}

impl Worker {
    pub fn new(pull_from: String, push_to: String) -> Result<Worker, Box<dyn Error>> {
        let context = zmq::Context::new();
        let pull_sock = context.socket(zmq::PULL)?;
        let push_sock = context.socket(zmq::PUSH)?;

        pull_sock.connect(&pull_from)?;
        push_sock.connect(&push_to)?;

        Ok(Worker {
            id: uuid::Uuid::new_v4(),
            pull_sock,
            push_sock,
        })
    }

    pub fn unpack(socket: &zmq::Socket) -> Result<(String, Vec<u8>), Box<dyn Error>> {
        let msgb = socket.recv_multipart(zmq::DONTWAIT)?;
        let cmd = &msgb[0];
        let data = &msgb[1];

        let cmd = String::from_utf8(cmd.to_vec())?;
        let data = data.to_vec();

        Ok((cmd, data))
    }

    pub fn push_frame(&self, cmd: &str, data: impl Serialize) -> Result<(), Box<dyn Error>> {
        let id = self.id.as_bytes();
        let cmd = cmd.as_bytes();
        let data = bincode::serialize(&data)?;

        self.push_sock.send_multipart([id, cmd, &data], 0)?;

        Ok(())
    }

    pub fn backprop(bpf: BackpropFrame) -> GradientFrame {
        let shape = bpf.ssm.shape();
        let size = shape[0];

        let mut weight_gradient = Array2::<f64>::zeros((shape[0], shape[1]));

        for neurons in bpf.neurons.iter() {
            let mut delta = Array1::<f64>::zeros(size);
            let nfpt = neurons
                .iter()
                .map(|nf| nf as *const NeuronFrame)
                .collect::<Vec<*const NeuronFrame>>();

            for neuron_ix in bpf.output_neurons.iter() {
                let neuron = &neurons[*neuron_ix];
                let gradient = neuron.loss;
                delta[*neuron_ix] += gradient;
            }

            for (neuron_ix, ..) in neurons.iter().enumerate() {
                if delta[neuron_ix] == 0. {
                    continue;
                }

                let connections = bpf
                    .ssm
                    .row(neuron_ix)
                    .iter()
                    .enumerate()
                    .map(|(other_neuron_ix, weight)| {
                        (other_neuron_ix, nfpt[other_neuron_ix], *weight)
                    })
                    .collect::<Vec<(usize, *const NeuronFrame, f64)>>();

                unsafe {
                    for (other_neuron_ix, other_neuron, weight) in connections {
                        if weight == 0. {
                            continue;
                        }

                        let grad = delta[neuron_ix] * Neuron::dsigmoid((*other_neuron).state);
                        weight_gradient[[neuron_ix, other_neuron_ix]] += grad;
                        delta[other_neuron_ix] +=
                            delta[neuron_ix] * weight * Neuron::dsigmoid((*other_neuron).state)
                    }
                }
            }
        }

        GradientFrame {
            step: bpf.step,
            gradient: weight_gradient,
        }
    }

    pub fn go(&self) {
        let prefix = format!("(powernet worker {})", self.id);

        let mut socks = [self.pull_sock.as_poll_item(zmq::POLLIN)];

        loop {
            if let Err(e) = zmq::poll(&mut socks, -1) {
                eprintln!("{prefix} failed to poll: {e}")
            }

            if socks[0].is_readable() {
                loop {
                    let (cmd, data) = match Worker::unpack(&self.pull_sock) {
                        Ok(d) => d,
                        Err(_) => break,
                    };

                    match cmd.as_str() {
                        "backprop" => {
                            let bpf: BackpropFrame = match bincode::deserialize(&data) {
                                Ok(frame) => frame,
                                Err(e) => {
                                    eprintln!("{prefix} failed to deserialize backprop frame: {e}");
                                    continue;
                                }
                            };

                            let gradient = Worker::backprop(bpf);

                            match self.push_frame("backprop", gradient) {
                                Err(e) => eprintln!("{prefix} failed to send gradients: {e}"),
                                _ => (),
                            }
                        }
                        "kill" => {
                            let _ = self.push_frame("kill", 0);
                        }
                        _ => eprintln!("{prefix} unknown command {}", cmd),
                    }
                }
            }
        }
    }
}

pub struct WorkerPool {
    pub desired: usize,
    pub push_to: String,
    pub pull_from: String,
    pub workers: Vec<JoinHandle<()>>,
    pub push_sock: zmq::Socket,
    pub pull_sock: zmq::Socket,
}

impl WorkerPool {
    pub fn new(up: String, down: String, n: usize) -> Result<WorkerPool, Box<dyn Error>> {
        let context = zmq::Context::new();
        let push_sock = context.socket(zmq::PUSH)?;
        let pull_sock = context.socket(zmq::PULL)?;

        push_sock.bind(&up)?;
        pull_sock.bind(&down)?;

        let workers = (0..n)
            .map(|_| WorkerPool::spawn_worker(up.clone(), down.clone()))
            .collect::<Vec<JoinHandle<()>>>();

        Ok(WorkerPool {
            desired: n,
            push_to: up,
            pull_from: down,
            workers,
            push_sock,
            pull_sock,
        })
    }

    pub fn spawn_worker(up: String, down: String) -> JoinHandle<()> {
        thread::spawn(move || {
            if let Ok(worker) = Worker::new(up, down) {
                worker.go()
            }
        })
    }

    pub fn send_frame(&self, cmd: &str, data: impl Serialize) -> Result<(), Box<dyn Error>> {
        let cmd = cmd.as_bytes();
        let data = bincode::serialize(&data)?;

        self.push_sock.send_multipart([cmd, &data], 0)?;

        Ok(())
    }

    pub fn unpack(&self) -> Result<(String, Vec<u8>), Box<dyn Error>> {
        let msgb = self.pull_sock.recv_multipart(zmq::DONTWAIT)?;

        let cmd = &msgb[1];
        let data = &msgb[2];

        let cmd = String::from_utf8(cmd.to_vec())?;
        let data = data.to_vec();

        Ok((cmd, data))
    }

    pub fn receive_frames(&mut self, expect: usize, timeout: f64) -> Vec<GradientFrame> {
        let mut received = 0;
        let cutoff = get_ts() + timeout;

        let mut socks = [self.pull_sock.as_poll_item(zmq::POLLIN)];

        let mut collected_gradients: Vec<GradientFrame> = vec![];

        while received < expect && get_ts() < cutoff {
            if let Err(e) = zmq::poll(&mut socks, 0) {
                eprintln!("(worker pool) failed to poll workers: {e}");
                continue;
            }

            loop {
                let (cmd, data) = match self.unpack() {
                    Ok(d) => d,
                    Err(_) => break,
                };

                match cmd.as_str() {
                    "backprop" => {
                        let frame: GradientFrame = match bincode::deserialize(&data) {
                            Ok(d) => d,
                            Err(e) => {
                                eprintln!("(worker pool) failed to deserialize gradient: {e}");
                                continue;
                            }
                        };

                        collected_gradients.push(frame);
                        received += 1;
                    }
                    _ => eprintln!("(worker pool) received unknown command from worker: {cmd}"),
                }
            }
        }

        collected_gradients
    }

    pub fn up(&mut self) -> Result<(), Box<dyn Error>> {
        self.workers.sort_by(|a, b| {
            if a.is_finished() && !b.is_finished() {
                return Ordering::Less;
            }
            if !a.is_finished() && !b.is_finished() {
                return Ordering::Greater;
            }
            return Ordering::Equal;
        });

        let mut active_workers = self.workers.len();

        for i in (0..active_workers).rev() {
            if self.workers[i].is_finished() {
                let _ = self.workers.pop();
                active_workers -= 1;
            } else {
                break;
            }
        }

        let spawn_n = self.desired - active_workers;

        let mut new_workers = (0..spawn_n)
            .map(|_| WorkerPool::spawn_worker(self.push_to.clone(), self.pull_from.clone()))
            .collect::<Vec<JoinHandle<()>>>();

        self.workers.append(&mut new_workers);

        if spawn_n > 0 {
            println!("(worker pool) revived {spawn_n} dead workers")
        }

        Ok(())
    }

    pub fn down(&mut self) -> Result<(), Box<dyn Error>> {
        while self.workers.len() > 0 {
            for _ in 0..self.workers.len() {
                if let Err(e) = self.send_frame("kill", 0) {
                    eprintln!("(worker pool) failed to send kill frame: {e}")
                }
            }

            self.workers.sort_by(|a, b| {
                if a.is_finished() && !b.is_finished() {
                    return Ordering::Less;
                }
                if !a.is_finished() && !b.is_finished() {
                    return Ordering::Greater;
                }
                return Ordering::Equal;
            });

            for i in (0..self.workers.len()).rev() {
                if self.workers[i].is_finished() {
                    match self.workers.pop() {
                        Some(w) => drop(w.join()),
                        _ => (),
                    }
                } else {
                    break;
                }
            }

            if self.workers.len() > 0 {
                println!(
                    "(worker pool) waiting on {} workers to exit",
                    self.workers.len()
                );
            } else {
                break;
            }

            thread::sleep(Duration::from_secs_f64(0.1));
        }

        Ok(())
    }
}

pub enum NeuronType {
    Input,
    Hidden,
    Output,
}

pub struct Neuron {
    pub state: f64,
    pub tau: f64,
    pub neuron_type: NeuronType,
}

impl Neuron {
    pub fn new() -> Neuron {
        Neuron {
            state: 0.,
            tau: 0.,
            neuron_type: NeuronType::Hidden,
        }
    }

    pub fn set_output(&mut self) -> &mut Self {
        self.neuron_type = NeuronType::Output;
        self
    }

    pub fn set_input(&mut self) -> &mut Self {
        self.neuron_type = NeuronType::Input;
        self
    }

    pub fn set_hidden(&mut self) -> &mut Self {
        self.neuron_type = NeuronType::Hidden;
        self
    }

    pub fn sigmoid(x: f64) -> f64 {
        let clipped_x = f64::max(f64::min(500., x), -500.);
        1. / (1. + f64::exp(-clipped_x))
    }

    pub fn dsigmoid(x: f64) -> f64 {
        Neuron::sigmoid(x) * (1. - Neuron::sigmoid(x))
    }

    pub fn next_state(
        &self,
        connections: Vec<(*const Neuron, f64)>,
        external_influence: f64,
    ) -> f64 {
        unsafe {
            -self.state
                + (connections
                    .iter()
                    .filter(|(_, w)| *w != 0.0)
                    .map(|(neuron, weight)| Neuron::sigmoid((**neuron).state) * weight)
                    .sum::<f64>())
                + external_influence
        }
    }

    pub fn euler_step(
        &mut self,
        connections: Vec<(*const Neuron, f64)>,
        next_tau: f64,
        external_influence: f64,
    ) -> (f64, f64) {
        let step_size = next_tau - self.tau;
        self.state += step_size * self.next_state(connections, external_influence);
        self.tau = next_tau;

        return (self.state, self.tau);
    }
}

pub struct PowerNetwork {
    pub size: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub density: f64,
    pub desired_connections: f64,

    pub neurons: Vec<Neuron>,
    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,
    pub weights: Array2<f64>,

    pub cache: VecDeque<Vec<NeuronFrame>>,
    pub trend: VecDeque<Vec<f64>>,
    pub trial: VecDeque<(usize, usize)>,
    pub timeout: VecDeque<(usize, usize, f64)>,

    pub tau: f64,
    pub steps: usize,
    pub recent_euler_steps: usize,

    pub initialized: bool,

    pub worker_pool: WorkerPool,
}

impl PowerNetwork {
    pub fn new(
        size: usize,
        d_in: usize,
        d_out: usize,
        workers: usize,
    ) -> Result<PowerNetwork, Box<dyn Error>> {
        let up: String = "tcp://127.0.0.1:3600".into();
        let down: String = "tcp://127.0.0.1:3601".into();

        let p = PowerNetwork {
            size,
            d_in,
            d_out,
            density: 0.,
            desired_connections: 0.,

            neurons: (0..size).map(|_| Neuron::new()).collect::<Vec<Neuron>>(),
            input_neurons: Vec::new(),
            output_neurons: Vec::new(),
            weights: Array2::zeros((size, size)),

            cache: VecDeque::new(),

            trend: VecDeque::new(),
            trial: VecDeque::new(),
            timeout: VecDeque::new(),

            tau: 0.,
            steps: 0,
            recent_euler_steps: 0,

            initialized: false,

            worker_pool: WorkerPool::new(up, down, workers)?,
        };

        Ok(p)
    }

    pub fn init_weight(&self, bound: Option<f64>, rng: &mut ThreadRng) -> f64 {
        if let Some(bound) = bound {
            rng.gen_range(-bound..bound)
        } else {
            let bound = f64::sqrt(6.) / (self.d_in + self.d_out) as f64;
            rng.gen_range(-bound..bound)
        }
    }

    pub fn init(&mut self, inputs: Vec<f64>, next_tau: f64) {
        if self.initialized == false {
            self.tau = next_tau;

            for neuron in self.neurons.iter_mut() {
                neuron.tau = next_tau;
            }

            for neuron_ix in self.output_neurons.iter() {
                let output_neuron = &mut self.neurons[*neuron_ix];
                output_neuron.neuron_type = NeuronType::Output;
            }

            for (i, neuron_ix) in self.input_neurons.iter().enumerate() {
                let input_neuron = &mut self.neurons[*neuron_ix];
                input_neuron.state = inputs[i];
                input_neuron.neuron_type = NeuronType::Input;
            }

            self.initialized = true;
        }
    }

    fn add_cache(&mut self, step_cache: Vec<NeuronFrame>) {
        self.cache.push_back(step_cache);
    }

    pub fn get_tau(&self) -> f64 {
        return self.tau;
    }

    pub fn weave(&mut self, density: f64) {
        self.density = density;
        let mut rng = thread_rng();

        let max_connections = self.size.pow(2);
        let mut desired_connections = (max_connections as f64 * self.density).floor() as usize;

        self.neurons = (0..self.size)
            .map(|_| Neuron::new())
            .collect::<Vec<Neuron>>();

        let ix_matrix = (0..self.size)
            .map(|y| {
                (0..self.size)
                    .map(move |x| (y, x))
                    .collect::<Vec<(usize, usize)>>()
            })
            .collect::<Vec<Vec<(usize, usize)>>>();

        let mut flat_ixlist = Vec::new();
        ix_matrix
            .into_iter()
            .for_each(|mut row| flat_ixlist.append(&mut row));
        flat_ixlist.shuffle(&mut rng);

        while desired_connections > 0 {
            let (y, x) = match flat_ixlist.pop() {
                Some(ix) => ix,
                None => continue,
            };

            self.weights[[y, x]] = self.init_weight(None, &mut rng);
            desired_connections -= 1;
        }

        let mut neuron_ixlist = (0..self.size).collect::<Vec<usize>>();
        neuron_ixlist.shuffle(&mut rng);

        self.input_neurons = neuron_ixlist.drain(0..self.d_in).collect::<Vec<usize>>();
        self.output_neurons = neuron_ixlist.drain(0..self.d_out).collect::<Vec<usize>>();
    }

    pub fn forward(&mut self, inputs: Vec<f64>, next_tau: f64, step_size: f64) -> Option<Vec<f64>> {
        if !self.initialized {
            self.init(inputs, next_tau);
            return None;
        }

        let mut steps = 0;

        while self.tau < next_tau {
            let mut input_ix = 0;
            steps += 1;

            let neuron_pointers = self
                .neurons
                .iter()
                .map(|n| n as *const Neuron)
                .collect::<Vec<*const Neuron>>();

            let mut step_cache: Vec<NeuronFrame> = Vec::new();

            for (neuron_ix, neuron) in self.neurons.iter_mut().enumerate() {
                let connections = self
                    .weights
                    .row(neuron_ix)
                    .iter()
                    .enumerate()
                    .map(|(other_neuron_ix, weight)| {
                        (neuron_pointers[other_neuron_ix] as *const Neuron, *weight)
                    })
                    .collect::<Vec<(*const Neuron, f64)>>();

                let state;
                let tau;
                let loss = 0.;

                match neuron.neuron_type {
                    NeuronType::Input => {
                        (state, tau) = neuron.euler_step(connections, self.tau, inputs[input_ix]);
                        input_ix += 1;
                    }
                    _ => {
                        (state, tau) = neuron.euler_step(connections, self.tau, 0.);
                    }
                }

                step_cache.push(NeuronFrame { state, loss, tau })
            }

            self.add_cache(step_cache);

            self.tau += step_size;
        }

        self.recent_euler_steps = steps;

        let output_state = self
            .neurons
            .iter()
            .filter(|n| match n.neuron_type {
                NeuronType::Output => true,
                _ => false,
            })
            .map(|n| n.state)
            .collect::<Vec<f64>>();

        Some(output_state)
    }

    pub fn backward(&mut self, learning_rate: f64, losses: Vec<f64>) {
        let steps = self.cache.len();
        self.trend.push_back(losses.clone());

        let inject_loss_start = steps - self.recent_euler_steps;
        let inject_loss_end = steps;

        for i in inject_loss_start..inject_loss_end {
            let cached_neurons = &mut self.cache[i];
            let mut output_ix = 0;
            for output_neuron_ix in self.output_neurons.iter() {
                cached_neurons[*output_neuron_ix].loss = losses[output_ix];
                output_ix += 1;
            }
        }

        let mut weight_gradient = Array2::<f64>::zeros((self.size, self.size));
        let workers = self.worker_pool.workers.len();

        let mut neuron_frames_per_core: Vec<Vec<Vec<NeuronFrame>>> = (0..workers)
            .map(|_| vec![])
            .collect::<Vec<Vec<Vec<NeuronFrame>>>>(
        );
        let mut round_robin = 0;

        for i in 0..steps {
            let core_ix = round_robin % workers;
            neuron_frames_per_core[core_ix].push(self.cache[i].clone());
            round_robin += 1;
        }

        let frames = neuron_frames_per_core
            .into_iter()
            .map(|nfpc| BackpropFrame {
                neurons: nfpc,
                step: 0,
                output_neurons: self.output_neurons.clone(),
                ssm: self.weights.clone(),
            })
            .collect::<Vec<BackpropFrame>>();

        frames.iter().for_each(|f| {
            if let Err(e) = self.worker_pool.send_frame("backprop", f) {
                println!("(power) pool invocation failed: {e}")
            }
        });

        let frames = self.worker_pool.receive_frames(frames.len(), 100.);
        frames.iter().for_each(|gradient_t| {
            weight_gradient += &gradient_t.gradient;
        });

        weight_gradient *= learning_rate;
        self.weights -= &weight_gradient;
    }

    pub fn metatrend(&self) -> f64 {
        let mut diffs: Vec<f64> = Vec::new();

        for i in 1..self.trend.len() {
            diffs.push(self.trend[i].iter().sum::<f64>() / self.trend[i - 1].iter().sum::<f64>());
        }

        return diffs.iter().sum::<f64>() / diffs.len() as f64;
    }

    pub fn alter(&mut self, alter_potential: usize) {
        if self.trend.len() < alter_potential {
            return;
        }

        if self.metatrend() > 1. {
            // Restore network alterations in case of poor metatrend

            if let Some(trial) = self.trial.pop_front() {
                self.weights[[trial.0, trial.1]] = 0.;
            }

            if let Some(timeout) = self.timeout.pop_front() {
                self.weights[[timeout.0, timeout.1]] = timeout.2;
            }
        }

        let mut rng = thread_rng();

        let mut output = self.output_neurons.clone();
        output.shuffle(&mut rng);

        let mut trial: Option<(usize, usize)> = None;
        let mut timeout: Option<(usize, usize)> = None;

        for output_neuron_ix in output {
            let mut output_weights = (0..self.size).collect::<Vec<_>>();
            output_weights.shuffle(&mut rng);

            for output_weight_ix in output_weights {
                if let None = timeout {
                    if self.weights[[output_neuron_ix, output_weight_ix]] > 0. {
                        timeout = Some((output_neuron_ix, output_weight_ix));
                    }
                }

                if let None = trial {
                    if self.weights[[output_neuron_ix, output_weight_ix]] <= 0. {
                        trial = Some((output_neuron_ix, output_weight_ix))
                    }
                }

                match (&trial, &timeout) {
                    (Some(_), Some(_)) => break,
                    _ => continue,
                }
            }
        }

        if let Some(trial) = trial {
            let trial_weight = self.init_weight(None, &mut rng);
            self.weights[[trial.0, trial.1]] = trial_weight;
            self.trial.push_back(trial);
        }

        if let Some(timeout) = timeout {
            let timeout_weight = self.weights[[timeout.0, timeout.1]];
            self.weights[[timeout.0, timeout.1]] = 0.;
            self.timeout
                .push_back((timeout.0, timeout.1, timeout_weight))
        }
    }

    pub fn sync(&mut self, retain: usize) {
        while self.cache.len() > retain {
            self.cache.pop_front();
        }

        drop(self.worker_pool.up());
    }
}
