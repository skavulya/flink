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

/** Represents loss functions which can be used with the [[GenericLossFunction]].
  *
  */
trait PartialLossFunction extends Serializable {
  /** Calculates the loss depending on the label and the prediction
    *
    * @param prediction
    * @param label
    * @return
    */
  def loss(prediction: Double, label: Double): Double

  /** Calculates the derivative of the [[PartialLossFunction]]
    * 
    * @param prediction
    * @param label
    * @return
    */
  def derivative(prediction: Double, label: Double): Double
}


/** Squared loss function which can be used with the [[GenericLossFunction]]
  *
  * The [[SquaredLoss]] function implements `1/2 (prediction - label)^2`
  */
object SquaredLoss extends PartialLossFunction {

  /** Calculates the loss depending on the label and the prediction
    *
    * @param prediction
    * @param label
    * @return
    */
  override def loss(prediction: Double, label: Double): Double = {
    0.5 * (prediction - label) * (prediction - label)
  }

  /** Calculates the derivative of the [[PartialLossFunction]]
    *
    * @param prediction
    * @param label
    * @return
    */
  override def derivative(prediction: Double, label: Double): Double = {
    (prediction - label)
  }
}


object LogisticLoss extends PartialLossFunction {
  /** Calculates the loss for a given prediction/truth pair
    *
    * @param prediction The predicted value
    * @param label The true value
    */
  override def loss(prediction: Double, label: Double): Double = {
    val t = prediction * label
    t match {
      case t if t > 18 => return math.exp(-t)
      case t if t < -18 => return -t
    }
    math.log(1 + math.exp(-t))
  }

  /** Calculates the derivative of the loss function with respect to the prediction
    *
    * @param prediction The predicted value
    * @param label The true value
    */
  override def derivative(prediction: Double, label: Double): Double = {
    (-label * math.exp(-label * prediction)) / (1 + math.exp(-label * prediction))
  }
}

class HingeLoss extends PartialLossFunction {
  /** Calculates the loss for a given prediction/truth pair
    *
    * @param prediction The predicted value
    * @param label The true value
    */
  override def loss(prediction: Double, label: Double): Double = {

    math.max(0, 1 - prediction * label)
  }

  /** Calculates the derivative of the loss function with respect to the prediction
    *
    * @param prediction The predicted value
    * @param label The true value
    */
  override def derivative(prediction: Double, label: Double): Double = {
    if (label * prediction < 1)
      -label * prediction
    else {
      0
    }
  }
}