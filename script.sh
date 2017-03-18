# Run a specific task by uncommenting corresponding line 

# Run Facescrub
# python ./discogan/image_translation.py --task_name='facescrub' --batch_size=500

# Run CelebA (blond to black hair, only female)
#python ./discogan/image_translation.py --task_name='celebA' --style_A='Blond_Hair' --style_B='Black_Hair' --constraint='Male' --constraint_type='-1' --batch_size=500

# Run CelebA (male to female)
python ./discogan/image_translation.py --task_name='celebA' --style_A='Male' --batch_size=500

# Run Edges2Handbags
# python ./discogan/image_translation.py --task_name='edges2handbags' --batch_size=500

# Run Edges2Shoes
# python ./discogan/image_translation.py --task_name='edges2shoes' --batch_size=500

# Run Shoes2Handbags
# python ./discogan/image_translation.py --task_name='shoes2handbags' --starting_rate=0.5 --batch_size=500

# Run Handbags2Shoes
# python ./discogan/image_translation.py --task_name='handbags2shoes' --batch_size=500

# Run Car2Car
#python ./discogan/angle_pairing.py --task_name='car2car' --batch_size=500

# Run Face2Face
# python ./discogan/angle_pairing.py --task_name='face2face' --batch_size=500

# Run Chair2Car
# python ./discogan/angle_pairing.py --task_name='chair2car' --batch_size=500

# Run Chair2Face
# python ./discogan/angle_pairing.py --task_name='chair2face' --batch_size=500

# Run Car2Face
# python ./discogan/angle_pairing.py --task_name='car2face' --batch_size=500
