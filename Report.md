# Project 2 Report

## Introduction
In this project, an agent is trained to move a double-joined arm to target locations under the  [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.
It is trained using the Deep Deterministic Policy Gradient (DDPG) algorithm. The environment contains 20 identical agents, each with its own copy of the environment.

## Learning Algorithm
Deep Deterministic Policy Gradient (DDPG) algorithm. is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data to learn the Q-function, and uses the Q-function to learn the policy. 
