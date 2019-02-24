package MyApp::Controller::Process;
use Mojo::Base 'Mojolicious::Controller';

sub upload_image {
	my $self = shift;

	# saving the image
	my $file = $self->param('the_image');
	my $directory = $self->uploads.generate_name();
	unless(-e $directory or mkdir $directory) {
		die "Unable to create $directory\n";
	}
	my $filepath = $directory.'/'.$file->filename;
	$file->move_to($filepath);

	# preprocessing
	my $script_to_run = $self->scripts . "test.py";
	my $output_path = `python3 $script_to_run $filepath`;

	# testing the image
	$script_to_run = $self->scripts . "steg_detect.py";
	my $results = `python3 $script_to_run $output_path`;
	my @results = split(/\n/, $results);

	# returning results
	$self->render(text => "success! @results");
}

sub upload_images {
  my $self = shift;
  $self->render(text => 'success2!');
}

sub generate_name {
	my @chars = qw/a b c d e f g h i j k l m n o p q r s t u v w x y z/;

	my $string = '';

	while (length($string) < 32) {
		my $choice = rand(1);
		if (($choice - 0) < (1 - $choice)) {
			$string .= $chars[int(rand(1)*25)];
		} else {
			$string .= int(rand(1)*9);
		}
	}
	return $string;
}

1;
