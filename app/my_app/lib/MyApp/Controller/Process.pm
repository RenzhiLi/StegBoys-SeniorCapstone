package MyApp::Controller::Process;
use Mojo::Base 'Mojolicious::Controller';
use Archive::Zip qw( :ERROR_CODES :CONSTANTS );
use Archive::Extract;

sub upload_image {
	my $self = shift;

	# saving the image
	my $file = $self->param('the_image');
	my $rand_name = generate_name();
	my $directory = $self->uploads.$rand_name."/";
	unless(-e $directory or mkdir $directory) {
		die "Unable to create $directory\n";
	}
	my $filepath = $directory.$file->filename;
	$file->move_to($filepath);

	# preprocessing
	my $script_to_run = $self->scripts . "Preprocessing.py";
	my $target_path = $directory."preprocessed.csv";
	`python3 $script_to_run $filepath $target_path '(64,64)'`;

	# testing the image
	$script_to_run = $self->scripts . "steg_detect.py";
	my $results = `python3 $script_to_run $target_path`;
	my @results = split(/\n/, $results);

	# writing good images to zip

	my $good_files = '';
	for (my $i = 0; $i < ((scalar @results) - 1); $i++) {
		$good_files .= $directory.$results[$i]."|";
	}

	$good_files .= $results[-1];

	my $output_zip = $directory."clean.zip";
	
	my $zip = Archive::Zip->new();

	$zip->addTreeMatching($directory, '', $good_files);

	$zip->writeToFileNamed($output_zip);

	# returning results
	$self->res->headers->content_type('application/zip');
	$self->reply->asset(Mojo::Asset::File->new(path => $output_zip));
}

sub upload_images {
	my $self = shift;

	# saving the image
	my $file = $self->param('the_images');
	my $rand_name = generate_name();
	my $directory = $self->uploads.$rand_name."/";
	unless(-e $directory or mkdir $directory) {
		die "Unable to create $directory\n";
	}
	my $filepath = $directory.$file->filename;
	$file->move_to($filepath);

	# unzip the archive
	my $ae = Archive::Extract->new( archive => $filepath );
    ### extract to /tmp ###

	$filepath = $directory."input_images/";
    my $ok = $ae->extract(to => $filepath);

	# preprocessing
	my $script_to_run = $self->scripts . "Preprocessing.py";
	my $target_path = $directory."preprocessed.csv";
	`python3 $script_to_run $filepath $target_path '(64,64)'`;

	# testing the image
	$script_to_run = $self->scripts . "steg_detect.py";
	my $results = `python3 $script_to_run $target_path`;
	my @results = split(/\n/, $results);

	# writing good images to zip

	my $good_files = '';
	for (my $i = 0; $i < ((scalar @results) - 1); $i++) {
		$good_files .= $directory.$results[$i]."|";
	}

	$good_files .= $results[-1];

	my $output_zip = $directory."clean.zip";
	
	my $zip = Archive::Zip->new();

	$zip->addTreeMatching($directory, '', $good_files);

	$zip->writeToFileNamed($output_zip);

	# returning results
	$self->res->headers->content_type('application/zip');
	$self->reply->asset(Mojo::Asset::File->new(path => $output_zip));
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
