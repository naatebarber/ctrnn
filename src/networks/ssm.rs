use ndarray::{Array1, Array2};
use rand::{prelude::*, thread_rng, Rng};
use std::collections::VecDeque;

use super::traits::ContinuousNetwork;

pub enum NeuronType {
    Input,
    Hidden,
    Output,
}

pub struct Neuron {
    pub state: f64,
    pub tau: f64,
    pub states: VecDeque<f64>,
    pub taus: VecDeque<f64>,
    pub targets: VecDeque<f64>,

    pub neuron_type: NeuronType,
}

impl Neuron {
    pub fn new() -> Neuron {
        Neuron {
            state: 0.,
            tau: 0.,
            states: VecDeque::new(),
            taus: VecDeque::new(),
            targets: VecDeque::new(),

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
    ) {
        let step_size = next_tau - self.tau;
        self.state += step_size * self.next_state(connections, external_influence);
        self.tau = next_tau;

        self.states.push_back(self.state);
        self.taus.push_back(self.tau);
    }

    pub fn cache_target(&mut self, target: f64) -> &mut Self {
        self.targets.push_back(target);
        self
    }

    pub fn sync<T>(a: &mut VecDeque<T>, b: &mut VecDeque<T>) {
        let (longer, shorter) = match a.len() > b.len() {
            true => (a, b),
            false => (b, a),
        };

        while longer.len() > shorter.len() {
            longer.pop_front();
        }
    }

    pub fn drain(&mut self, retain: usize) -> usize {
        if self.taus.len() != self.states.len() {
            println!("(tiny) states:taus mismatch, attempting sync...");
            Neuron::sync(&mut self.taus, &mut self.states)
        }

        if self.targets.len() > 0 {
            if self.targets.len() != self.states.len() {
                println!("(tiny) targets:(states:taus) mismatch, attempting sync...");
                Neuron::sync(&mut self.targets, &mut self.states);
                Neuron::sync(&mut self.targets, &mut self.taus);
            }
        }

        while self.states.len() > retain {
            self.states.pop_front();
            self.taus.pop_front();

            if self.targets.len() > 0 {
                self.targets.pop_front();
            }
        }

        return self.states.len();
    }
}

pub struct SsmNetwork {
    pub size: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub density: f64,
    pub desired_connections: f64,

    pub neurons: Vec<Neuron>,
    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,
    pub weights: Array2<f64>,

    pub tau: f64,
    pub steps: usize,

    pub initialized: bool,
    pub lifecycle: usize,
}

impl SsmNetwork {
    pub fn new(size: usize, d_in: usize, d_out: usize) -> SsmNetwork {
        SsmNetwork {
            size,
            d_in,
            d_out,
            density: 0.,
            desired_connections: 0.,

            neurons: (0..size).map(|_| Neuron::new()).collect::<Vec<Neuron>>(),
            input_neurons: Vec::new(),
            output_neurons: Vec::new(),
            weights: Array2::zeros((size, size)),

            tau: 0.,
            steps: 0,

            initialized: false,
            lifecycle: 0,
        }
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
}

impl ContinuousNetwork for SsmNetwork {
    fn get_tau(&self) -> f64 {
        return self.tau;
    }

    fn weave(&mut self, density: f64) {
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

    fn forward(
        &mut self,
        inputs: Vec<f64>,
        next_tau: f64,
        steps: usize,
        targets: Option<Vec<f64>>,
    ) -> Option<Vec<f64>> {
        if !self.initialized {
            self.init(inputs, next_tau);
            return None;
        }

        let step_size = (next_tau - self.tau) / steps as f64;

        while self.tau < next_tau {
            let mut input_ix = 0;
            let mut target_ix = 0;

            let neuron_pointers = self
                .neurons
                .iter()
                .map(|n| n as *const Neuron)
                .collect::<Vec<*const Neuron>>();

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

                match neuron.neuron_type {
                    NeuronType::Hidden => {
                        neuron.euler_step(connections, self.tau, 0.);
                    }
                    NeuronType::Input => {
                        neuron.euler_step(connections, self.tau, inputs[input_ix]);
                        input_ix += 1;
                    }
                    NeuronType::Output => {
                        neuron.euler_step(connections, self.tau, 0.);
                        if let Some(targets) = &targets {
                            neuron.cache_target(targets[target_ix]);
                            target_ix += 1;
                        }
                    }
                }
            }

            self.tau += step_size;
        }

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

    fn backward(&mut self, steps: usize, learning_rate: f64) -> Option<f64> {
        let mut weight_gradient = Array2::<f64>::zeros((self.size, self.size));
        let mut losses = (0..self.size).map(|_| vec![]).collect::<Vec<Vec<f64>>>();

        let neuron_pointers = self
            .neurons
            .iter()
            .map(|n| n as *const Neuron)
            .collect::<Vec<*const Neuron>>();

        for t in (0..steps).rev() {
            let mut delta = Array1::<f64>::zeros(self.size);

            for neuron_ix in self.output_neurons.iter() {
                let neuron = &mut self.neurons[*neuron_ix];
                let error = neuron.states[t] - neuron.targets[t];
                losses[*neuron_ix].push(error.powi(2));

                let gradient = 2. * error;

                delta[*neuron_ix] += gradient;
            }

            for (neuron_ix, ..) in self.neurons.iter_mut().enumerate() {
                if delta[neuron_ix] == 0. {
                    continue;
                }

                let connections = self
                    .weights
                    .row(neuron_ix)
                    .iter()
                    .enumerate()
                    .map(|(other_neuron_ix, weight)| {
                        (
                            other_neuron_ix,
                            neuron_pointers[other_neuron_ix] as *const Neuron,
                            *weight,
                        )
                    })
                    .collect::<Vec<(usize, *const Neuron, f64)>>();

                unsafe {
                    for (other_neuron_ix, other_neuron, weight) in connections {
                        if weight == 0. {
                            continue;
                        }

                        let grad = delta[neuron_ix] * Neuron::dsigmoid((*other_neuron).states[t]);
                        weight_gradient[[neuron_ix, other_neuron_ix]] += grad;
                        delta[other_neuron_ix] +=
                            delta[neuron_ix] * weight * Neuron::dsigmoid((*other_neuron).states[t])
                    }
                }
            }
        }

        weight_gradient *= learning_rate;
        self.weights -= &weight_gradient;

        let mut all_losses: Vec<f64> = vec![];
        for mut neuron_loss in losses {
            all_losses.append(&mut neuron_loss);
        }
        let loss = all_losses.iter().sum::<f64>() / all_losses.len() as f64;

        Some(loss)
    }

    fn step(
        &mut self,
        inputs: Vec<f64>,
        next_tau: f64,
        steps: usize,
        targets: Vec<f64>,
        retain: usize,
        learning_rate: f64,
    ) -> Option<(Vec<f64>, f64)> {
        self.lifecycle += 1;

        let network_output = match self.forward(inputs, next_tau, steps, Some(targets)) {
            Some(os) => os,
            None => return None,
        };

        let cross_neuron_bptt_steps = self
            .neurons
            .iter_mut()
            .map(|neuron| neuron.drain(retain))
            .filter(|state_length| *state_length > 0)
            .collect::<Vec<usize>>();

        let reference_steps = &cross_neuron_bptt_steps[0];
        let state_synchronized = cross_neuron_bptt_steps
            .iter()
            .all(|e| *e == *reference_steps);

        if !state_synchronized {
            panic!("(tiny) neurons fell out of sync!");
        }

        let loss = match self.backward(*reference_steps, learning_rate) {
            Some(l) => l,
            None => return None,
        };

        return Some((network_output, loss));
    }
}
