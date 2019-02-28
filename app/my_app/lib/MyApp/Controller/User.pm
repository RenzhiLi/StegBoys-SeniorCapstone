package MyApp::Controller::User;
use Mojo::Base 'Mojolicious::Controller';

# This action will render a template
sub index {
  my $self = shift;
  $self->redirect_to('/home');
}

1;
