import { AppBar, Toolbar, Typography, Button } from '@mui/material';

function Navbar() {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Used Car Price Predictor
        </Typography>
        <Button color="inherit">Home</Button>
        <Button color="inherit">About</Button>
      </Toolbar>
    </AppBar>
  );
}