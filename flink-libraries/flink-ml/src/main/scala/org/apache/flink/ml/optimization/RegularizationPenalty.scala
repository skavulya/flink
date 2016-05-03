/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.optimization

import org.apache.flink.ml.math.{Vector, BLAS}

/** Represents a type of regularization penalty
  *
  * Regularization penalties are used to restrict the optimization problem to solutions with
  * certain desirable characteristics, such as sparsity for the L1 penalty, or penalizing large
  * weights for the L2 penalty.
  *
  * The regularization term, $R(w)$ is added to the objective function, $f(w) = L(w) + \lambda R(w)$
  * where $\lambda$ is the regularization parameter used to tune the amount of regularization
  * applied.
  */
trait RegularizationPenalty extends Serializable {

  /** Calculates the new weights based on the gradient and regularization penalty
    *
    * @param weightVector The weights to be updated
    * @param gradient The gradient according to which we will update the weights
    * @param regularizationConstant The regularization parameter to be applied in the case of L1
    *                     regularization
    * @param learningRate The effective step size for this iteration
    * @return Updated weights
    */
  def takeStep(
                weightVector: Vector,
                gradient: Vector,
                regularizationConstant: Double,
                learningRate: Double
                ): Vector

  //TODO(skavulya): Include method for adding regularization to loss value?
}


/** $L_2$ regularization penalty.
  *
  * Penalizes large weights, favoring solutions with more small weights rather than few large ones.
  *
  */
object L2Regularization extends RegularizationPenalty {

  /** Calculates the new weights based on the gradient and regularization penalty
    *
    * @param weightVector The weights to be updated
    * @param gradient The gradient according to which we will update the weights
    * @param regularizationConstant The regularization parameter to be applied in the case of L1
    *                     regularization
    * @param learningRate The effective step size for this iteration
    * @return Updated weights
    */
  override def takeStep(
                         weightVector: Vector,
                         gradient: Vector,
                         regularizationConstant: Double,
                         learningRate: Double)
  : Vector = {
    // add the gradient of the L2 regularization
    BLAS.axpy(regularizationConstant, weightVector, gradient)

    // update the weights according to the learning rate
    BLAS.axpy(-learningRate, gradient, weightVector)

    weightVector
  }
}

/** $L_1$ regularization penalty.
  *
  * The $L_1$ penalty can be used to drive a number of the solution coefficients to 0, thereby
  * producing sparse solutions.
  *
  */
object L1Regularization extends RegularizationPenalty {

  /** Calculates the new weights based on the gradient and regularization penalty
    *
    * @param weightVector The weights to be updated
    * @param gradient The gradient according to which we will update the weights
    * @param regularizationConstant The regularization parameter to be applied in the case of L1
    *                     regularization
    * @param learningRate The effective step size for this iteration
    * @return Updated weights
    */
  override def takeStep(
                         weightVector: Vector,
                         gradient: Vector,
                         regularizationConstant: Double,
                         learningRate: Double)
  : Vector = {
    // Update weight vector with gradient. L1 regularization has no gradient, the proximal operator
    // does the job.
    BLAS.axpy(-learningRate, gradient, weightVector)

    // Apply proximal operator (soft thresholding)
    val shrinkageVal = regularizationConstant * learningRate
    var i = 0
    while (i < weightVector.size) {
      val wi = weightVector(i)
      weightVector(i) = scala.math.signum(wi) *
        scala.math.max(0.0, scala.math.abs(wi) - shrinkageVal)
      i += 1
    }

    weightVector
  }
}

//TODO(skavulya): Keep DiffRegularizationPenalty?
/** Abstract class for regularization penalties that are differentiable
  *
  */
/*
abstract class DiffRegularizationPenalty extends RegularizationPenalty {

  /** Compute the regularized gradient loss for the given data.
    * The provided cumGradient is updated in place.
    *
    * @param weightVector The current weight vector
    * @param lossGradient The vector to which the gradient will be added to, in place.
    * @return The regularized loss. The gradient is updated in place.
    */
  def regularizedLossAndGradient(
      loss: Double,
      weightVector: FlinkVector,
      lossGradient: FlinkVector,
      regularizationParameter: Double) : Double ={
    val adjustedLoss = regLoss(loss, weightVector, regularizationParameter)
    regGradient(weightVector, lossGradient, regularizationParameter)

    adjustedLoss
  }

  /** Adds regularization gradient to the loss gradient. The gradient is updated in place **/
  def regGradient(
      weightVector: FlinkVector,
      lossGradient: FlinkVector,
      regularizationParameter: Double)
}
*/