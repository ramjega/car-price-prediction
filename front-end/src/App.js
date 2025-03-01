import React, { useState } from 'react';
import axios from 'axios';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Card,
  TextField,
  Container,
  Box,
  MenuItem,
  Grid,
} from '@mui/material';

// Import a car image (replace with your own image path)
import carImage from './car-image.jpg'; // Ensure you have a car image in your project

function App() {
  // State for form data
  const [formData, setFormData] = useState({
    year: '',
    motor_volume: '',
    running_km: '',
    model: '',
    motor_type: '',
    wheel: '',
    color: '',
    type: '',
    status: '',
  });

  // State for predicted price
  const [prediction, setPrediction] = useState(null);

  // State for loading and error
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Dropdown options
  const modelOptions = ['kia', 'nissan', 'hyundai', 'mercedes-benz', 'toyota'];
  const motorTypeOptions = ['petrol', 'gas', 'petrol and gas'];
  const wheelOptions = ['left', 'right'];
  const colorOptions = [
    'black',
    'white',
    'silver',
    'blue',
    'gray',
    'other',
    'brown',
    'red',
    'green',
    'orange',
    'cherry',
    'skyblue',
    'clove',
    'beige',
  ];
  const typeOptions = ['sedan', 'SUV', 'Universal', 'Coupe', 'hatchback'];
  const statusOptions = ['excellent', 'normal', 'good', 'crashed', 'new'];

  // Function to capitalize the first letter of a string
  const capitalizeFirstLetter = (str) => {
    return str.charAt(0).toUpperCase() + str.slice(1);
  };

  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };


  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {

      const response = await axios.post('http://localhost:5000/predict', formData);
      setPrediction(response.data.predicted_price);
    } catch (err) {
      setError('Failed to fetch prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Navigation Bar */}
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Car Price Predictor
          </Typography>
          <Button color="inherit">Home</Button>
          <Button color="inherit">About</Button>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container sx={{ mt: 4, mb: 4 }}>
        {/* Description Card (Full Width) */}
        <Card sx={{ mb: 4, p: 3, boxShadow: 3, borderRadius: 2, textAlign: 'center' }}>
          <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 2 }}>
            Welcome to the Used Car Price Predictor!
          </Typography>
          <Typography variant="body1" sx={{ color: 'text.secondary' }}>
            Get an instant price prediction for your used car by filling out the form. Whether you're buying or selling, we've got you covered!
          </Typography>
        </Card>

        {/* Image and Form Section */}
        <Grid container spacing={4}>
          {/* Image Section */}
          <Grid item xs={12} md={6}>
            <Box
              component="img"
              src={carImage}
              alt="Car"
              sx={{
                width: '100%',
                height: 'auto',
                borderRadius: 2,
                boxShadow: 3,
              }}
            />

            {/* Predicted Price Display */}
            {(prediction || error) && (
              <Card
                sx={{
                  mt: 3,
                  p: 3,
                  textAlign: 'center',
                  backgroundColor: prediction ? '#e8f5e9' : '#ffebee',
                  boxShadow: 3,
                  borderRadius: 2,
                }}
              >
                {prediction && (
                  <Typography variant="h5" sx={{ fontWeight: 'bold', color: '#2e7d32' }}>
                    Predicted Price: ${prediction.toLocaleString()}
                  </Typography>
                )}
                {error && (
                  <Typography variant="h6" sx={{ color: '#d32f2f' }}>
                    {error}
                  </Typography>
                )}
              </Card>
            )}
          </Grid>

          {/* Form Section */}
          <Grid item xs={12} md={6}>
            <Card sx={{ p: 3, boxShadow: 3, borderRadius: 2 }}>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
                Predict Car Price
              </Typography>
              <form onSubmit={handleSubmit}>
                <Grid container spacing={2}>
                  {/* Year */}
                  <Grid item xs={6}>
                    <TextField
                      label="Year"
                      name="year"
                      type="number"
                      fullWidth
                      value={formData.year}
                      onChange={handleInputChange}
                      required
                    />
                  </Grid>

                  {/* Motor Volume */}
                  <Grid item xs={6}>
                    <TextField
                      label="Motor Volume (L)"
                      name="motor_volume"
                      type="number"
                      fullWidth
                      value={formData.motor_volume}
                      onChange={handleInputChange}
                      required
                    />
                  </Grid>

                  {/* Running Kilometers */}
                  <Grid item xs={12}>
                    <TextField
                      label="Running Kilometers"
                      name="running_km"
                      type="number"
                      fullWidth
                      value={formData.running_km}
                      onChange={handleInputChange}
                      required
                    />
                  </Grid>

                  {/* Model */}
                  <Grid item xs={6}>
                    <TextField
                      label="Model"
                      name="model"
                      select
                      fullWidth
                      value={formData.model}
                      onChange={handleInputChange}
                      required
                    >
                      {modelOptions.map((option) => (
                        <MenuItem key={option} value={option}>
                          {capitalizeFirstLetter(option)}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>

                  {/* Motor Type */}
                  <Grid item xs={6}>
                    <TextField
                      label="Motor Type"
                      name="motor_type"
                      select
                      fullWidth
                      value={formData.motor_type}
                      onChange={handleInputChange}
                      required
                    >
                      {motorTypeOptions.map((option) => (
                        <MenuItem key={option} value={option}>
                          {capitalizeFirstLetter(option)}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>

                  {/* Wheel */}
                  <Grid item xs={6}>
                    <TextField
                      label="Wheel"
                      name="wheel"
                      select
                      fullWidth
                      value={formData.wheel}
                      onChange={handleInputChange}
                      required
                    >
                      {wheelOptions.map((option) => (
                        <MenuItem key={option} value={option}>
                          {capitalizeFirstLetter(option)}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>

                  {/* Color */}
                  <Grid item xs={6}>
                    <TextField
                      label="Color"
                      name="color"
                      select
                      fullWidth
                      value={formData.color}
                      onChange={handleInputChange}
                      required
                    >
                      {colorOptions.map((option) => (
                        <MenuItem key={option} value={option}>
                          {capitalizeFirstLetter(option)}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>

                  {/* Type */}
                  <Grid item xs={6}>
                    <TextField
                      label="Type"
                      name="type"
                      select
                      fullWidth
                      value={formData.type}
                      onChange={handleInputChange}
                      required
                    >
                      {typeOptions.map((option) => (
                        <MenuItem key={option} value={option}>
                          {capitalizeFirstLetter(option)}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>

                  {/* Status */}
                  <Grid item xs={6}>
                    <TextField
                      label="Status"
                      name="status"
                      select
                      fullWidth
                      value={formData.status}
                      onChange={handleInputChange}
                      required
                    >
                      {statusOptions.map((option) => (
                        <MenuItem key={option} value={option}>
                          {capitalizeFirstLetter(option)}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>

                {/* Submit Button */}
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  fullWidth
                  sx={{ mt: 3, py: 1.5, fontWeight: 'bold' }}
                  disabled={loading}
                >
                  {loading ? 'Predicting...' : 'Predict'}
                </Button>
              </form>
            </Card>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

export default App;