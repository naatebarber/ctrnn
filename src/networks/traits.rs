pub trait ContinuousNetwork {
    fn get_tau(&self) -> f64;
    fn weave(&mut self, density: f64);
    fn forward(
        &mut self,
        inputs: Vec<f64>,
        next_tau: f64,
        steps: usize,
        targets: Option<Vec<f64>>,
    ) -> Option<Vec<f64>>;
    fn backward(&mut self, steps: usize, learning_rate: f64) -> Option<f64>;
    fn step(
        &mut self,
        inputs: Vec<f64>,
        next_tau: f64,
        steps: usize,
        targets: Vec<f64>,
        retain: usize,
        learning_rate: f64,
    ) -> Option<(Vec<f64>, f64)>;
}

pub trait ContinuousUnsupervisedNetwork {
    fn get_tau(&self) -> f64;
    fn weave(&mut self, density: f64);
    fn forward(
        &mut self,
        inputs: Vec<f64>,
        next_tau: f64,
        steps: usize,
        targets: Option<Vec<f64>>,
    ) -> Option<Vec<f64>>;
    fn backward(&mut self, steps: usize, learning_rate: f64);
    fn rstep(
        &mut self,
        inputs: Vec<f64>,
        next_tau: f64,
        steps: usize,
        losses: Vec<f64>,
        retain: usize,
        learning_rate: f64,
    ) -> Option<Vec<f64>>;
}
