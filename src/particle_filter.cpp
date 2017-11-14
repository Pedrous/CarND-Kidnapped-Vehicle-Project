/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Initialize the number of particles according to the first measurement
// Set the distributions and means for theta angle and locations of x and y
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	num_particles = 10;
	is_initialized = true;
	
	default_random_engine generator;
	normal_distribution<double> pdf_x(x, std[0]);
	normal_distribution<double> pdf_y(y, std[1]);
	normal_distribution<double> pdf_theta(theta, std[2]); 
	
	Particle particle;
	particle.weight = 1;
	for (int i = 0; i < num_particles; ++i) {
		particle.id = i;
		particle.x = pdf_x(generator);
		particle.y = pdf_y(generator);
		particle.theta = pdf_theta(generator);
	
		particles.push_back(particle);
		weights.push_back(1);
	}
}

// Predict new state for each particle using the current velocity and location information
// Take into account the situations when turning rate of the car is zero
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	default_random_engine generator;
	normal_distribution<double> pdf_x(0, std_pos[0]);
	normal_distribution<double> pdf_y(0, std_pos[1]);
	normal_distribution<double> pdf_theta(0, std_pos[2]);
	
	if (yaw_rate == 0) {
		for (auto& particle : particles) {
		
			particle.x += velocity * delta_t * cos(particle.theta) + pdf_x(generator);
			particle.y += velocity * delta_t * sin(particle.theta) + pdf_y(generator);
			particle.theta += pdf_theta(generator);
		}
	}
	else {
		for (auto& particle : particles) {
			particle.x += velocity/yaw_rate * (sin(particle.theta + delta_t * yaw_rate) - sin(particle.theta)) + pdf_x(generator);
			particle.y += velocity/yaw_rate * (cos(particle.theta) - cos(particle.theta + delta_t * yaw_rate)) + pdf_y(generator);
			particle.theta += delta_t * yaw_rate + pdf_theta(generator);
		}
	}
}

// Associate the observations to the landmarks within range
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	double distance, min_dist;
	int i;
	for (auto& observation : observations) {
		min_dist = -1;
		i = 0;
		for (const auto& predict : predicted) {
			distance = dist(predict.x, predict.y, observation.x, observation.y);
			if ((min_dist < 0) || (distance < min_dist)) {
				min_dist = distance;
				observation.id = i;
			}
			i++;
		}

	}
}

// Update the weights for each particle using the observations
// Transform the observations into the map coordinates, Associate them to the landmarks and ultimately update weights 

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	double c = sqrt(2 * M_PI * std_landmark[0] * std_landmark[1]);
	int i = 0;
	for (auto& particle : particles) {
		std::vector<LandmarkObs> mapObservations, map_landmarks_in_range;
		LandmarkObs mapObservation;
		double a, b, distx, disty;
		
		// STEP1: Transform car observations to map coordinates with respect to particle coordinates
		for (const auto& observation : observations) {
			mapObservation.id = observation.id;
			mapObservation.x = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
			mapObservation.y = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);
			mapObservations.push_back(mapObservation);
		}
	
		// STEP2: Associate and take into account only the landmarks that are close enough
		LandmarkObs landmark;
		for (const auto& map_landmark : map_landmarks.landmark_list) {
			if ( sensor_range >= dist(particle.x, particle.y, double(map_landmark.x_f), double(map_landmark.y_f)) ) {
				landmark.id = map_landmark.id_i;
				landmark.x = map_landmark.x_f;
				landmark.y = map_landmark.y_f;
				map_landmarks_in_range.push_back(landmark);
			}
		}	
		dataAssociation(map_landmarks_in_range, mapObservations);
		
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		
		// STEP 3: Update weights
		double w = 1;
		for (auto& mapObservation : mapObservations) {
			distx = mapObservation.x - map_landmarks_in_range[mapObservation.id].x;
			disty = mapObservation.y - map_landmarks_in_range[mapObservation.id].y;
			a = distx * distx / ( 2 * std_landmark[0] * std_landmark[0] );
			b = disty * disty / ( 2 * std_landmark[1] * std_landmark[1] );
			w *= exp(-(a + b)) / c;
			associations.push_back(map_landmarks_in_range[mapObservation.id].id);
			sense_x.push_back(map_landmarks_in_range[mapObservation.id].x);  
			sense_y.push_back(map_landmarks_in_range[mapObservation.id].y);
		}
		SetAssociations(particle, associations, sense_x, sense_y);
		
		particle.weight = w;
		weights[i] = w;
		i++;
	}
}

// Resample the according to the new weights
void ParticleFilter::resample() {
	
	default_random_engine generator;
	discrete_distribution<int> index(weights.begin(), weights.end());
	
	vector<Particle> particles2;
	for (const auto& particle : particles) {
        int i = index(generator);
        particles2.push_back(particles[i]);
    }
    particles = particles2;
}

Particle ParticleFilter::SetAssociations(Particle& particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
