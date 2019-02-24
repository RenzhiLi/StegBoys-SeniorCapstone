package MyApp::Controller::Navigation;
use Mojo::Base 'Mojolicious::Controller';

# This action will render a template
sub home {
  my $self = shift;

  # Render template "navigation/home.html.ep"
  $self->render();
}

sub about {
  my $self = shift;
  $self->render();
}

sub contact {
  my $self = shift;
  $self->render();
}

1;
