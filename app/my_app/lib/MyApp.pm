package MyApp;
use Mojo::Base 'Mojolicious';

# This method will run once at server start
sub startup {
  my $self = shift;

  # Load configuration from hash returned by config file
  my $config = $self->plugin('Config');

  # Configure the application
  $self->secrets($config->{secrets});
  $self->helper(uploads => sub {return $config->{uploads_loc}});
  $self->helper(scripts => sub {return $config->{scripts_loc}});
  $self->helper(csv => sub {return $config->{csv_loc}});

  # Router
  my $r = $self->routes;

  # Normal route to controller
  $r->get('/')->to('user#index');
  $r->get('/home')->to('navigation#home')->name('home');
  $r->get('/about')->to('navigation#about')->name('about');
  $r->get('/contact')->to('navigation#contact');
  $r->post('/upload_image')->to('process#upload_image');
  $r->post('/upload_images')->to('process#upload_images');
}

1;
