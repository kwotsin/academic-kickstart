git add .
git commit -m "Initial commit"
git push -u origin master

hugo
cd public
git add .
git commit -m "Build website"
git push origin master
cd ..