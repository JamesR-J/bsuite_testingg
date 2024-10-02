import chex
import jax
import jax.numpy as jnp
from rlax._src import distributions


Array = chex.Array


def policy_gradient_loss(
    logits_t: Array,
    a_t: Array,
    adv_t: Array,
    w_t: Array,
    uncertainty_t,
    use_stop_gradient: bool = True,
) -> Array:
  """Calculates the policy gradient loss.

  See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
  (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

  Args:
    logits_t: a sequence of unnormalized action preferences.
    a_t: a sequence of actions sampled from the preferences `logits_t`.
    adv_t: the observed or estimated advantages from executing actions `a_t`.
    w_t: a per timestep weighting for the loss.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.

  Returns:
    Loss whose gradient corresponds to a policy gradient update.
  """
  chex.assert_rank([logits_t, a_t, adv_t, w_t], [2, 1, 1, 1])
  chex.assert_type([logits_t, a_t, adv_t, w_t], [float, int, float, float])

  pi = distributions.softmax()
  log_pi_a_t = pi.logprob(a_t, logits_t)
  adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
  loss_per_timestep = - (log_pi_a_t * adv_t - (log_pi_a_t * pi))  # TODO is it in brackets idk?
  return jnp.mean(loss_per_timestep * w_t)